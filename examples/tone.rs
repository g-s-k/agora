use std::{
    f64::consts::TAU,
    io::{self, Write},
    sync::mpsc,
};

use agora::{Graph, Node};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize, SampleRate, StreamConfig,
};
use termion::{
    event::Key::{self, Char},
    input::TermRead,
    raw::IntoRawMode,
};

const BUF_SIZE: usize = 320;
const SAMPLE_RATE: usize = 48_000;

struct SineGen {
    current_phase: f64,
    frequency: f64,
}

impl Node<BUF_SIZE, SAMPLE_RATE> for SineGen {
    fn process(&mut self, buffer: &mut [f64; BUF_SIZE]) {
        for sample in buffer {
            self.current_phase += TAU * self.frequency / SAMPLE_RATE as f64;
            self.current_phase %= TAU;
            *sample = self.current_phase.sin();
        }
    }
}

struct SoftClip {
    gain: f64,
}

impl Node<BUF_SIZE, SAMPLE_RATE> for SoftClip {
    fn process(&mut self, buffer: &mut [f64; BUF_SIZE]) {
        for sample in buffer {
            *sample *= self.gain;
            *sample = sample.tanh();
        }
    }
}

enum Event {
    Freq0Change(f64),
    Freq1Change(f64),
    GainChange(f64),
}

fn main() {
    let (tx, rx) = mpsc::channel();
    let frequency_init = 220.0;

    let mut g = Graph::default();
    let node_0 = g.add_node(Box::new(SineGen {
        current_phase: 0.0,
        frequency: frequency_init,
    }));
    let node_1 = g.add_node(Box::new(SineGen {
        current_phase: 0.0,
        frequency: frequency_init,
    }));
    let node_2 = g.add_node(Box::new(SoftClip { gain: 1.0 }));

    g.connect(node_0, node_2).unwrap();
    g.connect(node_1, node_2).unwrap();

    let host = cpal::default_host();
    let output_device = host.default_output_device().unwrap();
    let config = output_device.default_output_config().unwrap();
    let channels = config.channels();

    let output_stream = output_device
        .build_output_stream(
            &StreamConfig {
                channels,
                sample_rate: SampleRate(SAMPLE_RATE as u32),
                buffer_size: BufferSize::Fixed(BUF_SIZE as u32),
            },
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                match rx.try_recv() {
                    Ok(Event::Freq0Change(frequency)) => {
                        g.replace_node(
                            node_0,
                            Box::new(SineGen {
                                current_phase: 0.0,
                                frequency,
                            }),
                        );
                    }
                    Ok(Event::Freq1Change(frequency)) => {
                        g.replace_node(
                            node_1,
                            Box::new(SineGen {
                                current_phase: 0.0,
                                frequency,
                            }),
                        );
                    }
                    Ok(Event::GainChange(gain)) => {
                        g.replace_node(node_2, Box::new(SoftClip { gain }));
                    }
                    _ => (),
                }

                let mut buf = [0.0; BUF_SIZE];
                g.process(&mut buf);

                for (frame, buf_sample) in data.chunks_mut(channels as usize).zip(buf) {
                    let value = cpal::Sample::from::<f32>(&(buf_sample as f32));
                    for sample in frame.iter_mut() {
                        *sample = value;
                    }
                }
            },
            |err| eprintln!("an error occurred on stream: {}", err),
        )
        .unwrap();

    output_stream.play().unwrap();

    let mut gain = 1.0;
    let mut freq_0 = frequency_init;
    let mut freq_1 = frequency_init;

    let mut stdout = io::stdout().into_raw_mode().unwrap();
    write!(stdout, "J:\tfrequency up 10Hz\tK:\tfrequency down 10Hz\r\n").unwrap();
    write!(
        stdout,
        "H:\tpitch up 1 semitone\tL:\tpitch down 1 semitone\r\n"
    )
    .unwrap();
    write!(
        stdout,
        "Up:\tfrequency up 10Hz\tDown:\tfrequency down 10Hz\r\n"
    )
    .unwrap();
    write!(
        stdout,
        "Left:\tpitch up 1 semitone\tRight:\tpitch down 1 semitone\r\n"
    )
    .unwrap();
    write!(stdout, "D:\tincrease gain 10%\tF:\tdecrease gain 10%\r\n").unwrap();
    write!(stdout, "Q:\tquit\r\n").unwrap();
    write!(
        stdout,
        "\rFrequency A: {:0.3}\tFrequency B: {:0.3}\tGain: {:0.3}",
        freq_0, freq_1, gain
    )
    .unwrap();
    stdout.flush().unwrap();

    for evt in io::stdin().keys() {
        match evt.unwrap() {
            // meta
            Char('q') | Key::Esc => return,
            // params
            Char('d') => {
                gain *= 0.9;
                tx.send(Event::GainChange(gain)).unwrap();
            }
            Char('f') => {
                gain *= 1.1;
                tx.send(Event::GainChange(gain)).unwrap();
            }
            Char('j') => {
                freq_0 -= 10.0;
                tx.send(Event::Freq0Change(freq_0)).unwrap();
            }
            Char('k') => {
                freq_0 += 10.0;
                tx.send(Event::Freq0Change(freq_0)).unwrap();
            }
            Char('h') => {
                freq_0 /= (2_f64).powf(1.0 / 12.0);
                tx.send(Event::Freq0Change(freq_0)).unwrap();
            }
            Char('l') => {
                freq_0 *= (2_f64).powf(1.0 / 12.0);
                tx.send(Event::Freq0Change(freq_0)).unwrap();
            }
            Key::Down => {
                freq_1 -= 10.0;
                tx.send(Event::Freq1Change(freq_1)).unwrap();
            }
            Key::Up => {
                freq_1 += 10.0;
                tx.send(Event::Freq1Change(freq_1)).unwrap();
            }
            Key::Left => {
                freq_1 /= (2_f64).powf(1.0 / 12.0);
                tx.send(Event::Freq1Change(freq_1)).unwrap();
            }
            Key::Right => {
                freq_1 *= (2_f64).powf(1.0 / 12.0);
                tx.send(Event::Freq1Change(freq_1)).unwrap();
            }
            _ => (),
        }

        write!(
            stdout,
            "\rFrequency A: {:0.3}\tFrequency B: {:0.3}\tGain: {:0.3}",
            freq_0, freq_1, gain
        )
        .unwrap();
        stdout.flush().unwrap();
    }
}
