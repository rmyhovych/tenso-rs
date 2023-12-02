use std::{
    fs::{metadata, File},
    os::unix::fs::FileExt,
};

pub struct MnistEntry {
    image: Vec<f32>,
    label: usize,
}

impl MnistEntry {
    pub const IMAGE_WIDTH: usize = 28;
    const IMAGE_SIZE: usize = MnistEntry::IMAGE_WIDTH * MnistEntry::IMAGE_WIDTH;

    pub fn get_image(&self) -> &Vec<f32> {
        &self.image
    }

    pub fn get_label(&self) -> usize {
        self.label
    }
}

pub struct MnistDataset {
    entries: Vec<MnistEntry>,
}

impl MnistDataset {
    pub fn load(image_file: &str, label_file: &str) -> Self {
        let mut image_data: Vec<u8> = Self::read_binary_file(image_file);
        image_data.drain(0..16);

        let mut label_data: Vec<u8> = Self::read_binary_file(label_file);
        label_data.drain(0..8);

        let entries = (0..label_data.len())
            .map(|i| {
                let image_offset = i * MnistEntry::IMAGE_SIZE;
                let image = image_data[image_offset..(image_offset + MnistEntry::IMAGE_SIZE)]
                    .iter()
                    .map(|v| (*v as f32) / 255.0)
                    .collect();
                let label = label_data[i] as usize;

                MnistEntry { image, label }
            })
            .collect();

        Self { entries }
    }

    pub fn get_entries(&self) -> &Vec<MnistEntry> {
        &self.entries
    }

    fn read_binary_file(filename: &str) -> Vec<u8> {
        match File::open(filename) {
            Ok(file) => {
                let file_len = match metadata(filename) {
                    Ok(meta) => meta.len(),
                    Err(_) => 0,
                };

                let mut buffer: Vec<u8> = vec![0; file_len as usize];
                file.read_at(buffer.as_mut_slice(), 0)
                    .expect("Buffer Overflow");

                buffer
            }
            Err(_) => Vec::new(),
        }
    }
}
