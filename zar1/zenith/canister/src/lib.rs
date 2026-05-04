//! ZAR-1 Zenith canister.
//!
//! Receives encrypted inference requests, decrypts them, runs ONNX-based
//! inference via `candle-onnx` compiled to WASM, batches up to 1000 requests,
//! and submits a single aggregated Plonk proof on-chain via the
//! `pallet-zk-verifier` host.

use std::sync::Mutex;

use candle_core::{Device, Tensor};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BATCH_LIMIT: usize = 1000;
const MODEL_BYTES: &[u8] = include_bytes!("../../zar1.onnx");

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InferRequest {
    /// Caller-provided ciphertext (e.g., XChaCha20-Poly1305).
    pub ciphertext: Vec<u8>,
    /// Per-request nonce.
    pub nonce: [u8; 24],
    /// Symmetric key (in production: ECDH-derived shared secret).
    pub key: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InferResponse {
    pub output_logits_hash: [u8; 32],
    pub n_tokens: usize,
}

#[derive(Default)]
struct Batch {
    requests: Vec<InferRequest>,
    responses: Vec<InferResponse>,
    /// Concatenated transcript: H(req_i || resp_i) for i in batch.
    transcript: Vec<u8>,
}

static BATCH: Lazy<Mutex<Batch>> = Lazy::new(|| Mutex::new(Batch::default()));
static MODEL: Lazy<Mutex<Option<candle_onnx::onnx::ModelProto>>> =
    Lazy::new(|| Mutex::new(None));

fn load_model() -> Result<candle_onnx::onnx::ModelProto, String> {
    candle_onnx::onnx::ModelProto::decode(MODEL_BYTES)
        .map_err(|e| format!("ONNX decode failed: {e}"))
}

/// Decrypt a request payload (placeholder: production should use ChaChaPoly1305).
fn decrypt(req: &InferRequest) -> Result<Vec<u8>, String> {
    if req.ciphertext.len() < 16 {
        return Err("ciphertext too short".into());
    }
    // XOR-with-key placeholder; replace with ring/chacha20poly1305.
    let mut out = req.ciphertext.clone();
    for (i, b) in out.iter_mut().enumerate() {
        *b ^= req.key[i % req.key.len()] ^ req.nonce[i % req.nonce.len()];
    }
    Ok(out)
}

/// Run a single forward pass with the cached ONNX model.
fn run_inference(input_ids: &[i64]) -> Result<Vec<f32>, String> {
    let mut guard = MODEL.lock().map_err(|e| e.to_string())?;
    if guard.is_none() {
        *guard = Some(load_model()?);
    }
    let model = guard.as_ref().unwrap();

    let device = Device::Cpu;
    let len = input_ids.len();
    let input = Tensor::from_slice(input_ids, (1, len), &device)
        .map_err(|e| e.to_string())?;
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("input_ids".to_string(), input);

    let outputs = candle_onnx::simple_eval(model, inputs).map_err(|e| e.to_string())?;
    let logits = outputs
        .get("logits")
        .ok_or_else(|| "missing logits output".to_string())?;
    let flat: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| e.to_string())?
        .flatten_all()
        .map_err(|e| e.to_string())?
        .to_vec1()
        .map_err(|e| e.to_string())?;
    Ok(flat)
}

fn hash_logits(v: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

/// Public canister entrypoint: enqueue a request and (if batch full) flush.
#[no_mangle]
pub extern "C" fn enqueue(req_json_ptr: *const u8, len: usize) -> *mut u8 {
    let bytes = unsafe { std::slice::from_raw_parts(req_json_ptr, len) };
    let req: InferRequest = match serde_json::from_slice(bytes) {
        Ok(r) => r,
        Err(e) => return ret_string(format!("ERR: {e}")),
    };

    let plaintext = match decrypt(&req) {
        Ok(p) => p,
        Err(e) => return ret_string(format!("ERR decrypt: {e}")),
    };

    // Interpret first 8*N bytes of plaintext as i64 token ids.
    let n = plaintext.len() / 8;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&plaintext[i * 8..i * 8 + 8]);
        ids.push(i64::from_le_bytes(buf));
    }

    let logits = match run_inference(&ids) {
        Ok(l) => l,
        Err(e) => return ret_string(format!("ERR infer: {e}")),
    };
    let resp = InferResponse {
        output_logits_hash: hash_logits(&logits),
        n_tokens: n,
    };

    let mut batch = BATCH.lock().unwrap();
    let mut t = Sha256::new();
    t.update(&req.ciphertext);
    t.update(&resp.output_logits_hash);
    batch.transcript.extend_from_slice(&t.finalize());
    batch.requests.push(req);
    batch.responses.push(resp.clone());

    if batch.requests.len() >= BATCH_LIMIT {
        if let Err(e) = flush_batch(&mut batch) {
            return ret_string(format!("ERR flush: {e}"));
        }
    }

    ret_string(serde_json::to_string(&resp).unwrap())
}

fn ret_string(s: String) -> *mut u8 {
    let mut bytes = s.into_bytes();
    bytes.push(0);
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

/// Aggregate the batch into a single Plonk proof and post on-chain.
fn flush_batch(batch: &mut Batch) -> Result<(), String> {
    let aggregate_hash = {
        let mut h = Sha256::new();
        h.update(&batch.transcript);
        h.finalize()
    };

    #[cfg(feature = "zk")]
    {
        let proof = pallet_zk_verifier::prove_plonk(&aggregate_hash)
            .map_err(|e| format!("plonk prove: {e:?}"))?;
        pallet_zk_verifier::submit_proof(proof, &aggregate_hash)
            .map_err(|e| format!("submit: {e:?}"))?;
    }

    log::info!(
        "Flushed batch of {} requests; root={}",
        batch.requests.len(),
        hex::encode(&aggregate_hash)
    );

    batch.requests.clear();
    batch.responses.clear();
    batch.transcript.clear();
    Ok(())
}

/// Force-flush; used by deploy scripts and tests.
#[no_mangle]
pub extern "C" fn force_flush() -> i32 {
    let mut batch = BATCH.lock().unwrap();
    if batch.requests.is_empty() {
        return 0;
    }
    match flush_batch(&mut batch) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}
