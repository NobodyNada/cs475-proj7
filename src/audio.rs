use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedIn, Resampler};

pub struct Player {
    pub stream: cpal::Stream,
    pub sample_rate: cpal::SampleRate,
    pub sample_buffer: Arc<Mutex<Vec<f32>>>,
}

pub fn play() -> Result<Player> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no audio device available"))?;
    let config = device
        .supported_output_configs()?
        .max_by(cpal::SupportedStreamConfigRange::cmp_default_heuristics)
        .ok_or_else(|| anyhow!("no configuration available"))?;

    let default_sample_rate = cpal::SampleRate(44100);
    let sample_rate = if config.max_sample_rate() < default_sample_rate {
        config.max_sample_rate()
    } else if config.min_sample_rate() > default_sample_rate {
        config.min_sample_rate()
    } else {
        default_sample_rate
    };
    let channels = config.channels();

    let config = config.with_sample_rate(sample_rate).config();

    let buf = include_bytes!("../pearl.mp3");
    let mut decoder = minimp3::Decoder::new(std::io::Cursor::new(buf));
    let mut resampler: Option<FftFixedIn<f32>> = None;
    let mut last_frame_sample_rate: usize = 0;

    let mut deinterleaved_buf: [Vec<f32>; 2] = Default::default();
    let mut resampled_buf: [Vec<f32>; 2] = Default::default();
    let mut interleaved_buf = VecDeque::<f32>::new();

    let mut resampler_tmp: [Vec<f32>; 2] = Default::default();

    let sample_buffer: Arc<Mutex<Vec<f32>>> = Default::default();
    let sample_buffer_ = sample_buffer.clone();

    let chunk_size = 128;

    let stream = device.build_output_stream(
        &config,
        move |mut data: &mut [f32], _| {
            while !data.is_empty() {
                if !interleaved_buf.is_empty() {
                    // Copy interleaved samples to the output stream
                    match channels {
                        2 => {
                            // Stereo to stereo: easy.
                            let samples_to_copy = std::cmp::min(data.len(), interleaved_buf.len());
                            std::iter::zip(
                                data.iter_mut(),
                                interleaved_buf.drain(0..samples_to_copy),
                            )
                            .for_each(|(t, s)| *t = s);
                            data = &mut data[samples_to_copy..];
                        }
                        1 => {
                            // Downmix stereo pairs to mono.
                            let samples_to_copy =
                                std::cmp::min(data.len(), interleaved_buf.len() / 2);
                            std::iter::zip(
                                data.iter_mut(),
                                interleaved_buf
                                    .drain(0..samples_to_copy)
                                    .scan(None, |prev, next| match prev.take() {
                                        Some(p) => {
                                            *prev = None;
                                            Some(Some((p + next) / 2.0))
                                        }
                                        None => {
                                            *prev = Some(next);
                                            Some(None)
                                        }
                                    })
                                    .flatten(),
                            )
                            .for_each(|(t, s)| *t = s);
                            data = &mut data[samples_to_copy..];
                        }
                        c => {
                            // It's some kind of surround device; copy stereo pairs to the first two output channels.
                            while !data.is_empty() && !interleaved_buf.is_empty() {
                                data[0] = interleaved_buf[0];
                                data[1] = interleaved_buf[1];
                                data = &mut data[c as usize..];
                                interleaved_buf.drain(0..2);
                            }
                        }
                    }
                } else if !resampled_buf[0].is_empty() {
                    // Interleave resampled samples

                    // But first, save them for the renderer
                    sample_buffer.lock().unwrap().extend(
                        std::iter::zip(resampled_buf[0].iter(), resampled_buf[1].iter())
                            .map(|(l, r)| (l + r) / 2.0),
                    );

                    let resampled = resampled_buf.split_at_mut(1);
                    interleaved_buf.extend(
                        std::iter::zip(resampled.0[0].drain(..), resampled.1[0].drain(..))
                            .flat_map(|(l, r)| [l, r].into_iter()),
                    );
                } else if deinterleaved_buf[0].len() > chunk_size {
                    // Resample deinterleaved samples
                    let resampler = resampler.get_or_insert_with(|| {
                        println!(
                            "Resampling from {last_frame_sample_rate} to {:#?}",
                            sample_rate.0
                        );
                        FftFixedIn::new(
                            last_frame_sample_rate,
                            sample_rate.0 as usize,
                            chunk_size,
                            2,
                            2,
                        )
                        .expect("failed to initialize resampler")
                    });
                    for (l, r) in std::iter::zip(
                        deinterleaved_buf[0].chunks_exact(chunk_size),
                        deinterleaved_buf[1].chunks_exact(chunk_size),
                    ) {
                        resampler
                            .process_into_buffer(&[l, r], &mut resampler_tmp, None)
                            .expect("resampling failed");
                        resampled_buf[0].append(&mut resampler_tmp[0]);
                        resampled_buf[1].append(&mut resampler_tmp[1]);
                    }
                    deinterleaved_buf[0].clear();
                    deinterleaved_buf[1].clear();
                } else {
                    // We need more data from the decoder!

                    match decoder.next_frame() {
                        Ok(f) => {
                            if f.sample_rate as usize != last_frame_sample_rate {
                                // We'll need a new resampler.
                                resampler = None;
                                last_frame_sample_rate = f.sample_rate as usize;
                            }

                            for samples in f.data.chunks(f.channels) {
                                deinterleaved_buf[0].push(samples[0] as f32 / i16::MAX as f32);
                                if f.channels > 1 {
                                    deinterleaved_buf[1].push(samples[1] as f32 / i16::MAX as f32);
                                } else {
                                    // just duplicate the mono channel into stereo
                                    deinterleaved_buf[1].push(samples[0] as f32 / i16::MAX as f32);
                                }
                            }
                        }
                        Err(minimp3::Error::Eof) => break,
                        Err(e) => {
                            panic!("Audio decode error: {:#?}", e)
                        }
                    }
                }
            }

            if !data.is_empty() {
                data.fill(0.0);
                sample_buffer
                    .lock()
                    .unwrap()
                    .extend(std::iter::repeat(0.0).take(data.len() / 2));
            }
        },
        move |err| {
            eprintln!("Audio playback error: {:#?}", err);
        },
    )?;

    stream.play()?;

    Ok(Player {
        stream,
        sample_rate,
        sample_buffer: sample_buffer_,
    })
}
