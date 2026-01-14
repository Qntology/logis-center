// use std::{fs::File, io::Write};

// use ffmpeg_next as ffmpeg;

// #[allow(unused)]
// fn save_file(
//     frame: &ffmpeg::frame::Video,
//     index: usize,
// ) -> std::result::Result<(), std::io::Error> {
//     let mut file = File::create(format!("frame{}.ppm", index))?;
//     file.write_all(format!("P6\n{} {}\n255\n", frame.width(), frame.height()).as_bytes())?;
//     file.write_all(frame.data(0))?;
//     Ok(())
// }
