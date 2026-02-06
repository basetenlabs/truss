use base64::{engine::general_purpose, Engine as _};
use numpy::PyArray1;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[derive(Clone)]
pub struct AudioProcessorConfig {
    pub sample_rate: u32,
    pub channels: u32,
    pub use_dynamic_normalization: bool,
    pub format: String,
    pub codec: String,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            use_dynamic_normalization: false,
            format: "f32le".to_string(),
            codec: "pcm_f32le".to_string(),
        }
    }
}

fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("Base64 decode failed: {}", e))
}

fn process_audio(audio_bytes: &[u8], config: &AudioProcessorConfig) -> Result<Vec<f32>, String> {
    let mut temp_file =
        NamedTempFile::new().map_err(|e| format!("Failed to create temp file: {}", e))?;
    temp_file
        .write_all(audio_bytes)
        .map_err(|e| format!("Failed to write to temp file: {}", e))?;
    temp_file
        .flush()
        .map_err(|e| format!("Failed to flush temp file: {}", e))?;

    let input_path = temp_file.path();

    let mut ffmpeg_cmd = Command::new("ffmpeg");
    ffmpeg_cmd
        .arg("-i")
        .arg(input_path)
        .arg("-f")
        .arg(&config.format)
        .arg("-acodec")
        .arg(&config.codec)
        .arg("-ac")
        .arg(config.channels.to_string())
        .arg("-ar")
        .arg(config.sample_rate.to_string())
        .arg("-");

    if config.use_dynamic_normalization {
        ffmpeg_cmd.args(["-af", "dynaudnorm"]);
    }

    ffmpeg_cmd.arg("-y"); // Overwrite output files without asking

    let output = ffmpeg_cmd
        .output()
        .map_err(|e| format!("ffmpeg execution failed: {}", e))?;
    // TODO: print ffmpeg command for debugging
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("ffmpeg command: {:?}", ffmpeg_cmd);

        return Err(format!("ffmpeg command line execution failed: {}", stderr));
    }

    let audio_data = output.stdout;
    let samples: Vec<f32> = audio_data
        .chunks(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0, 0, 0, 0]);
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok(samples)
}

#[pyclass]
pub struct MultimodalProcessor {
    client: Client,
    config: AudioProcessorConfig,
}

#[pymethods]
impl MultimodalProcessor {
    #[new]
    #[pyo3(signature = (
        sample_rate=16000,
        channels=1,
        use_dynamic_normalization=false,
        format="f32le",
        codec="pcm_f32le",
        timeout_secs=60
    ))]
    pub fn new(
        sample_rate: u32,
        channels: u32,
        use_dynamic_normalization: bool,
        format: &str,
        codec: &str,
        timeout_secs: u64,
    ) -> PyResult<Self> {
        let config = AudioProcessorConfig {
            sample_rate,
            channels,
            use_dynamic_normalization,
            format: format.to_string(),
            codec: codec.to_string(),
        };

        let mut client_builder = Client::builder();
        client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout_secs));

        let client = client_builder
            .build()
            .map_err(|e| PyException::new_err(format!("Failed to create client: {}", e)))?;

        Ok(Self { client, config })
    }

    #[pyo3(signature = (url, headers=None))]
    pub fn process_audio_from_url(
        &self,
        py: Python,
        url: String,
        headers: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let headers_map: Option<HashMap<String, String>> = if let Some(h) = headers {
            let mut map = HashMap::new();
            for (key, value) in h.iter() {
                let key_str = key.extract::<String>()?;
                let value_str = value.extract::<String>()?;
                map.insert(key_str, value_str);
            }
            Some(map)
        } else {
            None
        };

        let samples = py
            .allow_threads(|| {
                let mut request = self.client.get(&url);

                if let Some(ref hdrs) = headers_map {
                    for (key, value) in hdrs {
                        request = request.header(key, value);
                    }
                }

                let response = request
                    .send()
                    .map_err(|e| format!("Download failed: {}", e))?;

                let response = response
                    .error_for_status()
                    .map_err(|e| format!("HTTP error: {}", e))?;

                let audio_bytes = response
                    .bytes()
                    .map_err(|e| format!("Failed to read bytes: {}", e))?
                    .to_vec();

                process_audio(&audio_bytes, &self.config)
            })
            .map_err(|e| PyException::new_err(e))?;

        let numpy_array = PyArray1::from_vec(py, samples);
        Ok(numpy_array.into())
    }

    pub fn process_audio_from_base64(
        &self,
        py: Python,
        encoded: String,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let samples = py
            .allow_threads(|| {
                let audio_bytes = decode_base64(&encoded)?;

                process_audio(&audio_bytes, &self.config)
            })
            .map_err(|e| PyException::new_err(e))?;

        let numpy_array = PyArray1::from_vec(py, samples);
        Ok(numpy_array.into())
    }

    pub fn process_audio_from_bytes(
        &self,
        py: Python,
        audio_bytes: Vec<u8>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let samples = py
            .allow_threads(|| process_audio(&audio_bytes, &self.config))
            .map_err(|e| PyException::new_err(e))?;

        let numpy_array = PyArray1::from_vec(py, samples);
        Ok(numpy_array.into())
    }

    #[pyo3(signature = (url, headers=None))]
    pub fn download_bytes(
        &self,
        py: Python,
        url: String,
        headers: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyBytes>> {
        let headers_map: Option<HashMap<String, String>> = if let Some(h) = headers {
            let mut map = HashMap::new();
            for (key, value) in h.iter() {
                let key_str = key.extract::<String>()?;
                let value_str = value.extract::<String>()?;
                map.insert(key_str, value_str);
            }
            Some(map)
        } else {
            None
        };

        let bytes = py
            .allow_threads(|| {
                let mut request = self.client.get(&url);

                if let Some(ref hdrs) = headers_map {
                    for (key, value) in hdrs {
                        request = request.header(key, value);
                    }
                }

                request
                    .send()
                    .and_then(|resp| resp.bytes())
                    .map_err(|e| format!("Download failed: {}", e))
            })
            .map_err(|e| PyException::new_err(e))?;

        Ok(PyBytes::new(py, &bytes).into())
    }

    #[pyo3(signature = (source_type, source_data, headers=None))]
    pub fn process_audio(
        &self,
        py: Python,
        source_type: String,
        source_data: PyObject,
        headers: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        match source_type.as_str() {
            "url" => {
                let url = source_data.extract::<String>(py)?;
                self.process_audio_from_url(py, url, headers)
            }
            "base64" => {
                let encoded = source_data.extract::<String>(py)?;
                self.process_audio_from_base64(py, encoded)
            }
            "bytes" => {
                let bytes = source_data.extract::<Vec<u8>>(py)?;
                self.process_audio_from_bytes(py, bytes)
            }
            _ => Err(PyException::new_err(format!(
                "Unknown source type: {}",
                source_type
            ))),
        }
    }
}
