#![allow(clippy::needless_question_mark)]

use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[derive(Pod, Zeroable, Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Point {
    pub position: [f32; 4],
}
impl_vertex!(Point, position);

#[derive(Pod, Zeroable, Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Velocity {
    pub velocity: [f32; 3],
}

/// Our compute shader.
pub mod compute {
    /// The dimensions (along each axis) of one "cell" used for pressure computation.
    pub const PRESSURE_CELL_SIZE: f32 = 0.1;
    /// The total radius, centered on the origin, of the pressure simulation.
    /// The pressure is assumed to be 0 outside of this radius.)
    pub const PRESSURE_RADIUS: f32 = 20.0;
    /// The number of pressure cells along each axis.
    pub const NUM_CELLS_PERAXIS: u32 = (2.0 * PRESSURE_RADIUS / PRESSURE_CELL_SIZE) as u32;
    /// The total number of pressure cells.
    pub const NUM_CELLS_TOTAL: u32 = NUM_CELLS_PERAXIS.pow(3);
    /// The number of frequency bands.
    pub const EQ_BANDS: u32 = 64;
    impl SpecializationConstants {
        pub fn new() -> Self {
            Self {
                PRESSURE_RADIUS,
                PRESSURE_CELL_SIZE,
                NUM_CELLS_PERAXIS,
                NUM_CELLS_TOTAL,
                EQ_BANDS,
            }
        }
    }
    vulkano_shaders::shader! {
        ty: "compute",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
        src: "
        #version 450

        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        // Import our constants.
        layout(constant_id = 0) const float PRESSURE_CELL_SIZE = 1;
        layout(constant_id = 1) const float PRESSURE_RADIUS = 1;
        layout(constant_id = 2) const uint NUM_CELLS_PERAXIS = 1;
        layout(constant_id = 3) const uint NUM_CELLS_TOTAL = 1;
        layout(constant_id = 4) const uint EQ_BANDS = 1;

        const float G = 0.5;
        const float PRESSURE_STRENGTH = 1e-4;

        layout(set = 0, binding = 0) buffer Position {
            vec4 position[];   
        } position;
        layout(set = 0, binding = 1) buffer Velocity {
            vec3 velocity[];   
        } velocity;
        layout(set = 0, binding = 2) buffer Pressure {
           uint pressure[];   
        } pressure;
        layout(set = 1, binding = 0) uniform Uniforms {
            // Whether to write to the first or second half of the pressure buffer.
            bool which_pressure_buffer;
            // The average amplitude of the audio signal.
            float amplitude;
            // The amount of time since last time we ran the simulation.
            float elapsed;
        } uniforms;
        layout(set = 1, binding = 1) buffer Bands {
            vec4 bands[];
        } bands;

        // Hashing algorithm from Stack Overflow user Spatial:
        // https://stackoverflow.com/a/17479300/3476191
        uint hash( uint x ) {
            x += ( x << 10u );
            x ^= ( x >>  6u );
            x += ( x <<  3u );
            x ^= ( x >> 11u );
            x += ( x << 15u );
            return x;
        }
        uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
        uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
        uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

        // Construct a float with half-open range [0:1] using low 23 bits.
        // All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
        float floatConstruct( uint m ) {
            const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
            const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

            m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
            m |= ieeeOne;                          // Add fractional part to 1.0

            float  f = uintBitsToFloat( m );       // Range [1:2]
            return f - 1.0;                        // Range [0:1]
        }

        // Pseudo-random value in half-open range [0:1].
        float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
        float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
        float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
        float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }


        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 p = position.position[i].xyz / position.position[i].w;
            vec3 v = velocity.velocity[i];

            // Which frequency band are we?
            uint band_idx = i * EQ_BANDS / gl_NumWorkGroups.x * gl_WorkGroupSize.x;

            uint prev, next;
            if (band_idx != 0) prev = band_idx - 1;
            else prev = band_idx;
            if (band_idx != EQ_BANDS - 1) next = band_idx + 1;
            else next = band_idx;

            float frac = float(i % EQ_BANDS) / EQ_BANDS;

            // Compute the intensity of audio within this frequency band (interpolating to an
            // adjacent frequency band if we're close to the edge).
            float intensity;
            if (frac < 0.5) intensity = mix(bands.bands[prev].x, bands.bands[band_idx].x, frac*2);
            else intensity = mix(bands.bands[next].x, bands.bands[band_idx].x, (frac + 0.5) * 2);

            // Which pressure cell are we?
            int cell_x = int((p.x + PRESSURE_RADIUS) / PRESSURE_CELL_SIZE);
            int cell_y = int((p.y + PRESSURE_RADIUS) / PRESSURE_CELL_SIZE);
            int cell_z = int((p.z + PRESSURE_RADIUS) / PRESSURE_CELL_SIZE);

            int cell;
            if      (cell_x < 0 || cell_x > NUM_CELLS_PERAXIS) cell = -1;
            else if (cell_y < 0 || cell_y > NUM_CELLS_PERAXIS) cell = -1;
            else if (cell_z < 0 || cell_z > NUM_CELLS_PERAXIS) cell = -1;
            else cell = int((cell_z*NUM_CELLS_PERAXIS + cell_y)*NUM_CELLS_PERAXIS + cell_x);

            // Gravity is towards the origin.
            v -= normalize(p) * G;
            // Drag will reduce velocity, but the force of drag is reduced as the
            // amplitude of our freqency band is increased.
            float drag = 1 - intensity;
            v *= 1 - drag * uniforms.elapsed;
            if (cell != -1) {
                // Pressure will add a random offset to our velocity, scaled by the number of
                // nearby particles and the overall audio amplitude.
                uint h = hash(floatBitsToUint(v)) ^ hash(floatBitsToUint(p));
                vec3 r = vec3(floatConstruct(hash(h)), floatConstruct(hash(h + 1)), floatConstruct(hash(h + 2)));
                r -= vec3(0.5, 0.5, 0.5);
                float p = pressure.pressure[cell + (uniforms.which_pressure_buffer ? NUM_CELLS_PERAXIS : 0)];
                v += r * PRESSURE_STRENGTH * pressure.pressure[cell] * uniforms.amplitude * uniforms.elapsed;
            }
            // Update position.
            p.xyz += v * uniforms.elapsed;

            position.position[i] = vec4(p, 1);
            velocity.velocity[i] = v;

            // Update pressures for next frame: increment the number of particles in our cell.
            atomicAdd(
                pressure.pressure[cell + (uniforms.which_pressure_buffer ? 0 : NUM_CELLS_PERAXIS)],
                1
            );
        }
        "
    }
}

/// The vertex shader.
pub mod vertex {
    impl SpecializationConstants {
        pub fn new() -> Self {
            Self {
                EQ_BANDS: super::compute::EQ_BANDS,
            }
        }
    }
    vulkano_shaders::shader! {
        ty: "vertex",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
        src: "
        #version 450
        layout(location = 0) in vec4 position;

        layout(constant_id = 0) const uint EQ_BANDS = 1;

        layout(set = 0, binding = 0) uniform Uniforms {
            // The projection/view matrix.
            mat4 matrix;
            // The total numer of particles.
            uint num_points;
            bool magic;
        } uniforms;
        layout(set = 0, binding = 1) buffer Bands {
            vec4 bands[];
        } bands;

        layout(location = 0) out vec4 color;

        // https://stackoverflow.com/a/17897228/3476191
        // All components are in the range [0…1], including hue.
        vec3 hsv2rgb(vec3 c)
        {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            gl_Position = uniforms.matrix * position;

            // Which frequency band are we?
            uint band_idx = gl_VertexIndex * EQ_BANDS / uniforms.num_points;

            uint prev, next;
            if (band_idx != 0) prev = band_idx - 1;
            else prev = band_idx;
            if (band_idx != EQ_BANDS - 1) next = band_idx + 1;
            else next = band_idx;

            float frac = float(gl_VertexIndex % EQ_BANDS) / EQ_BANDS;

            float intensity;
            if (frac < 0.5) intensity = mix(bands.bands[prev].x, bands.bands[band_idx].x, frac*2);
            else intensity = mix(bands.bands[next].x, bands.bands[band_idx].x, (frac + 0.5) * 2);

            // Color the vertex such that the hue corresponds to the frequency of our band and the
            // brightness corresponds to the amplitude.
            float hue = float(gl_VertexIndex) / uniforms.num_points;
            color = vec4(hsv2rgb(vec3(hue, 1, 1)), 1) * intensity;

            gl_PointSize = 1 + (5 * 3) * int(uniforms.magic);
            color.a += float(uniforms.magic) * 42;
        }
        "
    }
}

/// The simplest fragment shader imaginable.
/// Usually.
pub mod fragment {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(location = 0) in vec4 in_color;
        layout(location = 0) out vec4 out_color;

        vec2 i_point_c = ivec2(gl_PointCoord * 5.);

        void main() {
            vec3 color = vec3(in_color);

            if (in_color.a > 42) {
                bool magic = 
                    i_point_c == ivec2(0, 0)
                    || i_point_c == ivec2(0, 4)
                    || i_point_c == ivec2(0, 3)
                    || i_point_c == ivec2(2, 4)
                    || i_point_c.x == 4;

                bool magic2 = i_point_c == ivec2(3, 1)
                    || i_point_c == ivec2(2, 1);

                if (magic) discard;
                color += vec3(magic2) * sqrt(sqrt(in_color.a - 42));
            }
        
            out_color = vec4(color, 1.0);
        }
        "
    }
}
