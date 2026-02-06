use base64::{engine::general_purpose, Engine as _};
use numpy::PyArray1;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;
use tempfile::NamedTempFile;

static FFMPEG_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn check_ffmpeg_available() -> bool {
    *FFMPEG_AVAILABLE.get_or_init(|| Command::new("ffmpeg").arg("-version").output().is_ok())
}

fn warn_if_ffmpeg_not_available() {
    if !check_ffmpeg_available() {
        eprintln!("Warning: ffmpeg is not installed or not in PATH. Audio processing features will not be available.");
    }
}

#[derive(Clone)]
pub struct AudioProcessorConfig {
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub use_dynamic_normalization: Option<bool>,
    pub format: String,
    pub codec: Option<String>,
    pub raw_ffmpeg_args: Vec<String>,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: None,
            channels: None,
            use_dynamic_normalization: None,
            format: "f32le".to_string(),
            codec: None,
            raw_ffmpeg_args: Vec::new(),
        }
    }
}

impl AudioProcessorConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct AudioConfig {
    #[pyo3(get, set)]
    pub sample_rate: Option<u32>,
    #[pyo3(get, set)]
    pub channels: Option<u32>,
    #[pyo3(get, set)]
    pub use_dynamic_normalization: Option<bool>,
    #[pyo3(get, set)]
    pub format: String,
    #[pyo3(get, set)]
    pub codec: Option<String>,
    #[pyo3(get, set)]
    pub raw_ffmpeg_args: Option<Vec<String>>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: None,
            channels: None,
            use_dynamic_normalization: None,
            format: "f32le".to_string(),
            codec: None,
            raw_ffmpeg_args: None,
        }
    }
}

#[pymethods]
impl AudioConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sample_rate(&self, sample_rate: u32) -> Self {
        let mut new_config = self.clone();
        new_config.sample_rate = Some(sample_rate);
        new_config
    }

    pub fn with_channels(&self, channels: u32) -> Self {
        let mut new_config = self.clone();
        new_config.channels = Some(channels);
        new_config
    }

    pub fn with_use_dynamic_normalization(&self, use_dynamic_normalization: bool) -> Self {
        let mut new_config = self.clone();
        new_config.use_dynamic_normalization = Some(use_dynamic_normalization);
        new_config
    }

    pub fn with_format(&self, format: String) -> Self {
        let mut new_config = self.clone();
        new_config.format = format;
        new_config
    }

    pub fn with_codec(&self, codec: String) -> Self {
        let mut new_config = self.clone();
        new_config.codec = Some(codec);
        new_config
    }

    pub fn with_raw_ffmpeg_args(&self, args: Vec<String>) -> Self {
        let mut new_config = self.clone();
        new_config.raw_ffmpeg_args = Some(args);
        new_config
    }
}

impl AudioConfig {
    pub fn build(&self) -> AudioProcessorConfig {
        AudioProcessorConfig {
            sample_rate: self.sample_rate,
            channels: self.channels,
            use_dynamic_normalization: self.use_dynamic_normalization,
            format: self.format.clone(),
            codec: self.codec.clone(),
            raw_ffmpeg_args: self.raw_ffmpeg_args.clone().unwrap_or_default(),
        }
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct Headers {
    #[pyo3(get, set)]
    pub headers: HashMap<String, String>,
}

#[pymethods]
impl Headers {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&self, key: String, value: String) -> Self {
        let mut new_headers = self.clone();
        new_headers.headers.insert(key, value);
        new_headers
    }
}

impl Headers {
    pub fn build(&self) -> HashMap<String, String> {
        self.headers.clone()
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct TimingInfo {
    #[pyo3(get, set)]
    pub total_us: f64,
    #[pyo3(get, set)]
    pub download_us: f64,
    #[pyo3(get, set)]
    pub processing_us: f64,
}

#[pymethods]
impl TimingInfo {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TimingInfo(total={:.0}µs, download={:.0}µs, processing={:.0}µs)",
            self.total_us, self.download_us, self.processing_us
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("Base64 decode failed: {}", e))
}

struct ProcessedAudio {
    samples: Vec<f32>,
    timing: TimingInfo,
}

fn process_audio(
    audio_bytes: &[u8],
    config: &AudioProcessorConfig,
) -> Result<ProcessedAudio, String> {
    let start = Instant::now();
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
    ffmpeg_cmd.arg("-i").arg(input_path);
    ffmpeg_cmd.arg("-f").arg(&config.format);

    if let Some(codec) = &config.codec {
        ffmpeg_cmd.arg("-acodec").arg(codec);
    }

    if let Some(channels) = config.channels {
        ffmpeg_cmd.arg("-ac").arg(channels.to_string());
    }

    if let Some(sample_rate) = config.sample_rate {
        ffmpeg_cmd.arg("-ar").arg(sample_rate.to_string());
    }

    if config.use_dynamic_normalization.unwrap_or(false) {
        ffmpeg_cmd.args(["-af", "dynaudnorm"]);
    }

    for arg in &config.raw_ffmpeg_args {
        ffmpeg_cmd.arg(arg);
    }

    ffmpeg_cmd.arg("-y").arg("-");

    let output = ffmpeg_cmd
        .output()
        .map_err(|e| format!("ffmpeg execution failed: {}", e))?;

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

    let processing_us = start.elapsed().as_secs_f64() * 1_000_000.0;

    Ok(ProcessedAudio {
        samples,
        timing: TimingInfo {
            total_us: processing_us,
            download_us: 0.0,
            processing_us,
        },
    })
}

#[pyclass]
pub struct MultimodalProcessor {
    #[pyo3(get, set)]
    pub timeout_secs: u64,
    async_client: reqwest::Client,
}

#[pymethods]
impl MultimodalProcessor {
    #[new]
    #[pyo3(signature = (timeout_secs=60))]
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            async_client: reqwest::Client::new(),
        }
    }

    #[pyo3(signature = (url, audio_config, /, headers=None))]
    pub fn process_audio_from_url<'py>(
        &self,
        py: Python<'py>,
        url: String,
        audio_config: Bound<'_, AudioConfig>,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let headers_map: Option<HashMap<String, String>> =
            headers.map(|headers_obj| headers_obj.borrow().build());

        let async_client = self.async_client.clone();

        let future = async move {
            let download_start = Instant::now();

            let mut request = async_client.get(&url);

            if let Some(ref hdrs) = headers_map {
                for (key, value) in hdrs {
                    request = request.header(key, value);
                }
            }

            let response = request
                .send()
                .await
                .map_err(|e| PyException::new_err(format!("Download failed: {}", e)))?;

            let response = response
                .error_for_status()
                .map_err(|e| PyException::new_err(format!("HTTP error: {}", e)))?;

            let audio_bytes = response
                .bytes()
                .await
                .map_err(|e| PyException::new_err(format!("Failed to read bytes: {}", e)))?;

            let download_us = download_start.elapsed().as_secs_f64() * 1_000_000.0;

            let processed = process_audio(&audio_bytes, &config)
                .map_err(|e| PyException::new_err(e))?;

            let timing = TimingInfo {
                total_us: processed.timing.total_us,
                download_us,
                processing_us: processed.timing.processing_us,
            };

            Python::with_gil(|py| {
                let numpy_array = PyArray1::from_vec(py, processed.samples);
                let timing_py = Py::new(py, timing)?;
                #[allow(deprecated)]
                let numpy_obj = numpy_array.into_py(py);
                #[allow(deprecated)]
                let timing_obj = timing_py.into_py(py);
                let tuple = (numpy_obj, timing_obj).into_pyobject(py)?;
                #[allow(deprecated)]
                Ok::<PyObject, PyErr>(tuple.into_py(py))
            })
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (encoded, audio_config))]
    pub fn process_audio_from_base64<'py>(
        &self,
        py: Python<'py>,
        encoded: String,
        audio_config: Bound<'_, AudioConfig>,
    ) -> PyResult<Bound<'py, PyAny>> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let future = async move {
            let audio_bytes = decode_base64(&encoded)
                .map_err(|e| PyException::new_err(e))?;

            let processed = process_audio(&audio_bytes, &config)
                .map_err(|e| PyException::new_err(e))?;

            Python::with_gil(|py| {
                let numpy_array = PyArray1::from_vec(py, processed.samples);
                let timing_py = Py::new(py, processed.timing)?;
                #[allow(deprecated)]
                let numpy_obj = numpy_array.into_py(py);
                #[allow(deprecated)]
                let timing_obj = timing_py.into_py(py);
                let tuple = (numpy_obj, timing_obj).into_pyobject(py)?;
                #[allow(deprecated)]
                Ok::<PyObject, PyErr>(tuple.into_py(py))
            })
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (audio_bytes, audio_config))]
    pub fn process_audio_from_bytes<'py>(
        &self,
        py: Python<'py>,
        audio_bytes: Vec<u8>,
        audio_config: Bound<'_, AudioConfig>,
    ) -> PyResult<Bound<'py, PyAny>> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let future = async move {
            let processed = process_audio(&audio_bytes, &config)
                .map_err(|e| PyException::new_err(e))?;

            Python::with_gil(|py| {
                let numpy_array = PyArray1::from_vec(py, processed.samples);
                let timing_py = Py::new(py, processed.timing)?;
                #[allow(deprecated)]
                let numpy_obj = numpy_array.into_py(py);
                #[allow(deprecated)]
                let timing_obj = timing_py.into_py(py);
                let tuple = (numpy_obj, timing_obj).into_pyobject(py)?;
                #[allow(deprecated)]
                Ok::<PyObject, PyErr>(tuple.into_py(py))
            })
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (url, /, headers=None))]
    pub fn download_bytes<'py>(
        &self,
        py: Python<'py>,
        url: String,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers_map: Option<HashMap<String, String>> =
            headers.map(|headers_obj| headers_obj.borrow().build());

        let async_client = self.async_client.clone();

        let future = async move {
            let mut request = async_client.get(&url);

            if let Some(ref hdrs) = headers_map {
                for (key, value) in hdrs {
                    request = request.header(key, value);
                }
            }

            let response = request
                .send()
                .await
                .map_err(|e| PyException::new_err(format!("Download failed: {}", e)))?;

            let response = response
                .error_for_status()
                .map_err(|e| PyException::new_err(format!("HTTP error: {}", e)))?;

            let bytes = response
                .bytes()
                .await
                .map_err(|e| PyException::new_err(format!("Failed to read bytes: {}", e)))?;

            Python::with_gil(|py| {
                let py_bytes = PyBytes::new(py, &bytes);
                Ok(py_bytes.unbind())
            })
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (source_type, source_data, audio_config, /, headers=None))]
    pub fn process_audio<'py>(
        &self,
        py: Python<'py>,
        source_type: String,
        source_data: PyObject,
        audio_config: Bound<'_, AudioConfig>,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        warn_if_ffmpeg_not_available();
        match source_type.as_str() {
            "url" => {
                let url = source_data.extract::<String>(py)?;
                self.process_audio_from_url(py, url, audio_config.clone(), headers)
            }
            "base64" => {
                let encoded = source_data.extract::<String>(py)?;
                self.process_audio_from_base64(py, encoded, audio_config.clone())
            }
            "bytes" => {
                let bytes = source_data.extract::<Vec<u8>>(py)?;
                self.process_audio_from_bytes(py, bytes, audio_config.clone())
            }
            _ => Err(PyException::new_err(format!(
                "Unknown source type: {}",
                source_type
            ))),
        }
    }
}
