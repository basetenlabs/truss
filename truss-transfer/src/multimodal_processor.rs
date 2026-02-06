use base64::{engine::general_purpose, Engine as _};
use numpy::PyArray1;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
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
    pub use_pipes: Option<bool>,
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
            use_pipes: None,
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
    #[pyo3(get, set)]
    pub use_pipes: Option<bool>,
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
            use_pipes: None,
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

    pub fn with_use_pipes(&self, use_pipes: bool) -> Self {
        let mut new_config = self.clone();
        new_config.use_pipes = Some(use_pipes);
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
            use_pipes: self.use_pipes,
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
    #[pyo3(get, set)]
    pub format_detection_us: f64,
    #[pyo3(get, set)]
    pub input_method: String,
}

#[pymethods]
impl TimingInfo {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TimingInfo(total={:.0}µs, download={:.0}µs, processing={:.0}µs, format_detection={:.0}µs, input_method={})",
            self.total_us, self.download_us, self.processing_us, self.format_detection_us, self.input_method
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

enum InputSource<'a> {
    Pipe,
    Path(&'a std::path::Path),
}

fn build_ffmpeg_command(input_source: InputSource, config: &AudioProcessorConfig) -> Command {
    let out_format = "f32le";
    let mut cmd = Command::new("ffmpeg");

    // Common flags
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");

    // Input source (only difference)
    match input_source {
        InputSource::Pipe => {
            // Don't use -nostdin for pipes - we need to write to stdin
        }
        InputSource::Path(_) => {
            // Use -nostdin for files to prevent interactive prompts
            cmd.arg("-nostdin");
        }
    };

    match input_source {
        InputSource::Pipe => cmd.arg("-i").arg("pipe:0"),
        InputSource::Path(path) => cmd.arg("-i").arg(path),
    };

    // Output format (same for both)
    cmd.arg("-f")
        .arg(out_format)
        .arg("-acodec")
        .arg("pcm_f32le");

    // Conditional args (same for both)
    if let Some(channels) = config.channels {
        cmd.arg("-ac").arg(channels.to_string());
    }
    if let Some(sample_rate) = config.sample_rate {
        cmd.arg("-ar").arg(sample_rate.to_string());
    }
    if config.use_dynamic_normalization.unwrap_or(false) {
        cmd.args(["-af", "dynaudnorm"]);
    }
    for arg in &config.raw_ffmpeg_args {
        cmd.arg(arg);
    }

    // Output to pipe (same for both)
    cmd.arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // For pipes, also set stdin to piped
    if matches!(input_source, InputSource::Pipe) {
        cmd.stdin(Stdio::piped());
    }

    cmd
}

fn should_use_pipes_heuristic(audio_bytes: &[u8]) -> bool {
    // Need at least 12 bytes for reliable detection
    if audio_bytes.len() < 12 {
        return false; // Too small, play safe with tempfile
    }

    let magic = &audio_bytes[..12.min(audio_bytes.len())];

    // MP3 (ID3v2 tag) - works with pipes
    if magic.starts_with(b"ID3") {
        return true;
    }

    // MP3 (raw, no ID3) - sync byte 0xFF followed by MPEG sync bits
    if magic[0] == 0xFF && (magic[1] & 0xE0) == 0xE0 {
        return true;
    }

    // WAV (RIFF) - works with pipes
    if magic.starts_with(b"RIFF") {
        return true;
    }

    // FLAC - works with pipes
    if magic.starts_with(b"fLaC") {
        return true;
    }

    // OGG (includes Opus, Vorbis, Speex) - works with pipes
    if magic.starts_with(b"OggS") {
        return true;
    }

    // ADTS AAC (raw AAC stream) - works with pipes
    // Sync byte 0xFF followed by 0xF0 (ADTS fixed header)
    if magic[0] == 0xFF && (magic[1] & 0xF0) == 0xF0 {
        return true;
    }

    // M4A/MP4 - NEEDS tempfile (ftyp atom at offset 4, moov atom at end)
    if audio_bytes.len() >= 8 && &audio_bytes[4..8] == b"ftyp" {
        return false;
    }

    // Unknown format - try pipes first (will fallback if fails)
    true
}

fn process_audio_with_pipes(
    audio_bytes: &[u8],
    config: &AudioProcessorConfig,
) -> Result<ProcessedAudio, String> {
    let start = Instant::now();
    let mut cmd = build_ffmpeg_command(InputSource::Pipe, config);

    // Spawn the process
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("ffmpeg spawn failed: {e}"))?;

    // Write to stdin
    let mut stdin = child.stdin.take().ok_or("ffmpeg stdin missing")?;
    stdin
        .write_all(audio_bytes)
        .map_err(|e| format!("stdin write failed: {e}"))?;
    drop(stdin); // Close stdin

    // Read output
    let mut stdout = child.stdout.take().ok_or("ffmpeg stdout missing")?;
    let mut stderr = child.stderr.take().ok_or("ffmpeg stderr missing")?;

    let mut out = Vec::new();
    stdout
        .read_to_end(&mut out)
        .map_err(|e| format!("stdout read failed: {e}"))?;

    let mut err_s = String::new();
    let _ = stderr.read_to_string(&mut err_s);

    let status = child.wait().map_err(|e| format!("wait failed: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg failed: {err_s}"));
    }

    if out.is_empty() {
        return Err("Empty output - format may not support pipes".to_string());
    }

    if out.len() % 4 != 0 {
        return Err(format!("Unexpected f32le output length: {}", out.len()));
    }

    let samples = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();

    let processing_us = start.elapsed().as_secs_f64() * 1_000_000.0;

    Ok(ProcessedAudio {
        samples,
        timing: TimingInfo {
            total_us: processing_us,
            download_us: 0.0,
            processing_us,
            format_detection_us: 0.0,
            input_method: "pipe".to_string(),
        },
    })
}

fn process_audio_with_tempfile(
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

    let mut cmd = build_ffmpeg_command(InputSource::Path(temp_file.path()), config);
    cmd.stdin(Stdio::piped()); // Set stdin for tempfile version

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("ffmpeg spawn failed: {e}"))?;

    let mut stdout = child.stdout.take().ok_or("ffmpeg stdout missing")?;
    let mut stderr = child.stderr.take().ok_or("ffmpeg stderr missing")?;

    let mut out = Vec::new();
    stdout
        .read_to_end(&mut out)
        .map_err(|e| format!("stdout read failed: {e}"))?;

    let mut err_s = String::new();
    let _ = stderr.read_to_string(&mut err_s);

    let status = child.wait().map_err(|e| format!("wait failed: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg failed: {err_s}"));
    }

    if out.len() % 4 != 0 {
        return Err(format!("Unexpected f32le output length: {}", out.len()));
    }

    let samples = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();

    let processing_us = start.elapsed().as_secs_f64() * 1_000_000.0;

    Ok(ProcessedAudio {
        samples,
        timing: TimingInfo {
            total_us: processing_us,
            download_us: 0.0,
            processing_us,
            format_detection_us: 0.0,
            input_method: "tempfile".to_string(),
        },
    })
}

fn process_audio(
    audio_bytes: &[u8],
    config: &AudioProcessorConfig,
) -> Result<ProcessedAudio, String> {
    let total_start = Instant::now();

    // Determine which method to use based on config
    let format_detection_start = Instant::now();
    let use_pipes = match config.use_pipes {
        Some(true) => true,                              // Force pipes
        Some(false) => false,                            // Force tempfile
        None => should_use_pipes_heuristic(audio_bytes), // Auto-detect
    };
    let format_detection_us = format_detection_start.elapsed().as_secs_f64() * 1_000_000.0;

    let result = if use_pipes {
        // Try pipes first, fallback to tempfile on failure
        match process_audio_with_pipes(audio_bytes, config) {
            Ok(mut processed) => {
                processed.timing.format_detection_us = format_detection_us;
                processed.timing.total_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
                Ok(processed)
            }
            Err(_e) => {
                // Fallback to tempfile if pipes fail
                eprint!("Pipes failed, falling back to tempfile. Please set use_pipes=False to avoid this fallback.\n");
                let mut processed = process_audio_with_tempfile(audio_bytes, config)?;
                processed.timing.format_detection_us = format_detection_us;
                processed.timing.total_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
                Ok(processed)
            }
        }
    } else {
        // Use tempfile directly
        let mut processed = process_audio_with_tempfile(audio_bytes, config)?;
        processed.timing.format_detection_us = format_detection_us;
        processed.timing.total_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
        Ok(processed)
    };

    result
}

#[pyclass]
pub struct MultimodalProcessor {
    client: Client,
}

#[pymethods]
impl MultimodalProcessor {
    #[new]
    #[pyo3(signature = (
        timeout_secs=300
    ))]
    pub fn new(timeout_secs: u64) -> PyResult<Self> {
        let mut client_builder = Client::builder();
        client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout_secs));

        let client = client_builder
            .build()
            .map_err(|e| PyException::new_err(format!("Failed to create client: {}", e)))?;

        Ok(Self { client })
    }

    #[pyo3(signature = (url, audio_config, /, headers=None))]
    pub fn process_audio_from_url(
        &self,
        py: Python,
        url: String,
        audio_config: Bound<'_, AudioConfig>,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<TimingInfo>)> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let headers_map: Option<HashMap<String, String>> =
            headers.map(|headers_obj| headers_obj.borrow().build());

        let (processed, download_us) = py
            .allow_threads(|| {
                let download_start = Instant::now();
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

                let download_us = download_start.elapsed().as_secs_f64() * 1_000_000.0;

                let processed = process_audio(&audio_bytes, &config)?;

                Ok((processed, download_us))
            })
            .map_err(|e: String| PyException::new_err(e))?;

        let timing = TimingInfo {
            total_us: download_us + processed.timing.total_us,
            download_us,
            processing_us: processed.timing.processing_us,
            format_detection_us: processed.timing.format_detection_us,
            input_method: processed.timing.input_method,
        };

        let numpy_array = PyArray1::from_vec(py, processed.samples);
        let timing_py = Py::new(py, timing)?;
        Ok((numpy_array.into(), timing_py))
    }

    #[pyo3(signature = (encoded, audio_config))]
    pub fn process_audio_from_base64(
        &self,
        py: Python,
        encoded: String,
        audio_config: Bound<'_, AudioConfig>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<TimingInfo>)> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let processed = py
            .allow_threads(|| {
                let audio_bytes = decode_base64(&encoded)?;

                process_audio(&audio_bytes, &config)
            })
            .map_err(|e: String| PyException::new_err(e))?;

        let numpy_array = PyArray1::from_vec(py, processed.samples);
        let timing_py = Py::new(py, processed.timing)?;
        Ok((numpy_array.into(), timing_py))
    }

    #[pyo3(signature = (audio_bytes, audio_config))]
    pub fn process_audio_from_bytes(
        &self,
        py: Python,
        audio_bytes: Vec<u8>,
        audio_config: Bound<'_, AudioConfig>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<TimingInfo>)> {
        warn_if_ffmpeg_not_available();
        let config = audio_config.borrow().build();

        let processed = py
            .allow_threads(|| process_audio(&audio_bytes, &config))
            .map_err(|e: String| PyException::new_err(e))?;

        let numpy_array = PyArray1::from_vec(py, processed.samples);
        let timing_py = Py::new(py, processed.timing)?;
        Ok((numpy_array.into(), timing_py))
    }

    #[pyo3(signature = (url, /, headers=None))]
    pub fn download_bytes(
        &self,
        py: Python,
        url: String,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<Py<PyBytes>> {
        let headers_map: Option<HashMap<String, String>> =
            headers.map(|headers_obj| headers_obj.borrow().build());

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
            .map_err(PyException::new_err)?;

        Ok(PyBytes::new(py, &bytes).into())
    }

    #[pyo3(signature = (source_type, source_data, audio_config, /, headers=None))]
    pub fn process_audio(
        &self,
        py: Python,
        source_type: String,
        source_data: PyObject,
        audio_config: Bound<'_, AudioConfig>,
        headers: Option<Bound<'_, Headers>>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<TimingInfo>)> {
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
