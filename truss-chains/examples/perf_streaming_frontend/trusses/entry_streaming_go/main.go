// Go WS entrypoint for the streaming front-end showcase.
//
// Step trace via log/slog. On WS connect: per \n-terminated FRAME|<spk>|<text>
// line, fan out a transcribe call to TranscriberMock (HTTP POST, in a
// goroutine). On WS close / DONE / FRAME|END sentinel: send full buffer to
// DiarizerMock (HTTP POST). Wait for all transcribes + diarize. Send both
// to AssignerMock (Rust). Emit merged JSON frame back to client.
//
// Sibling URLs: /etc/b10_dynamic_config/dynamic_chainlet_config
// Auth key:     /secrets/baseten_chain_api_key
// No chains SDK runs in this pod.

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	dynamicConfigPath = "/etc/b10_dynamic_config/dynamic_chainlet_config"
	apiKeyPath        = "/secrets/baseten_chain_api_key"
	port              = ":8000"
	endSentinel       = "FRAME|END"
)

type siblingCfg struct {
	Name        string          `json:"name"`
	PredictURL  string          `json:"predict_url"`
	InternalURL json.RawMessage `json:"internal_url"`
}

var (
	chainCfg map[string]siblingCfg
	apiKey   string
	logger   *slog.Logger
	upgrader = websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}
)

func loadChainCfg() error {
	b, err := os.ReadFile(dynamicConfigPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", dynamicConfigPath, err)
	}
	if err := json.Unmarshal(b, &chainCfg); err != nil {
		return fmt.Errorf("parse dynamic_chainlet_config: %w", err)
	}
	for name, c := range chainCfg {
		c.Name = name
		chainCfg[name] = c
	}
	return nil
}

func loadAPIKey() error {
	b, err := os.ReadFile(apiKeyPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", apiKeyPath, err)
	}
	apiKey = strings.TrimSpace(string(b))
	return nil
}

type siblingResult struct {
	Name   string      `json:"name"`
	Status int         `json:"status,omitempty"`
	Body   interface{} `json:"body,omitempty"`
	Error  string      `json:"error,omitempty"`
}

func callHTTPSibling(ctx context.Context, name string, payload map[string]interface{}) siblingResult {
	cfg, ok := chainCfg[name]
	if !ok {
		logger.Error("[STEP] sibling not in dynamic_chainlet_config", "sibling", name)
		return siblingResult{Name: name, Error: "sibling not in dynamic_chainlet_config"}
	}
	body, _ := json.Marshal(payload)
	logger.Info("[STEP] HTTP POST to sibling", "sibling", name, "url", cfg.PredictURL, "payload_bytes", len(body))
	req, err := http.NewRequestWithContext(ctx, "POST", cfg.PredictURL, bytes.NewReader(body))
	if err != nil {
		logger.Error("[STEP] new request failed", "sibling", name, "err", err)
		return siblingResult{Name: name, Error: err.Error()}
	}
	req.Header.Set("Authorization", "Api-Key "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 30 * time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("[STEP] sibling call failed", "sibling", name, "err", err)
		return siblingResult{Name: name, Error: err.Error()}
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var parsed interface{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		parsed = string(raw)
	}
	logger.Info("[STEP] sibling call returned", "sibling", name, "status", resp.StatusCode, "bytes", len(raw), "elapsed_ms", time.Since(start).Milliseconds())
	return siblingResult{Name: name, Status: resp.StatusCode, Body: parsed}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func handleStream(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		logger.Error("[WS] upgrade failed", "err", err)
		return
	}
	defer conn.Close()
	logger.Info("[WS] client connected", "remote", r.RemoteAddr)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var (
		mu              sync.Mutex
		incoming        bytes.Buffer
		residual        bytes.Buffer
		transcribeWG    sync.WaitGroup
		transcribeOrder int
		transcribeRes   []siblingResult
	)

	processCompletedLines := func() {
		raw := residual.Bytes()
		for {
			idx := bytes.IndexByte(raw, '\n')
			if idx < 0 {
				break
			}
			line := strings.TrimSpace(string(raw[:idx]))
			raw = raw[idx+1:]
			if line == "" || !strings.HasPrefix(line, "FRAME|") || line == endSentinel {
				continue
			}
			mu.Lock()
			transcribeOrder++
			order := transcribeOrder
			mu.Unlock()
			logger.Info("[STEP] frame received, fanning out transcribe", "order", order, "frame", line)
			transcribeWG.Add(1)
			go func(frame string, ord int) {
				defer transcribeWG.Done()
				res := callHTTPSibling(ctx, "TranscriberMock", map[string]interface{}{
					"frame": frame,
					"order": ord,
				})
				mu.Lock()
				transcribeRes = append(transcribeRes, res)
				mu.Unlock()
			}(line, order)
		}
		residual.Reset()
		residual.Write(raw)
	}

	for {
		mt, data, err := conn.ReadMessage()
		if err != nil {
			logger.Info("[WS] read ended", "err", err)
			break
		}
		if mt == websocket.TextMessage && string(data) == "DONE" {
			logger.Info("[WS] DONE received from client")
			break
		}
		if mt != websocket.BinaryMessage && mt != websocket.TextMessage {
			continue
		}
		logger.Info("[STEP] buffer_bytes", "chunk_bytes", len(data))
		incoming.Write(data)
		residual.Write(data)
		processCompletedLines()
		if bytes.Contains(data, []byte(endSentinel)) {
			logger.Info("[STEP] FRAME|END sentinel observed; stopping ingest")
			break
		}
	}

	fullBuffer := append([]byte{}, incoming.Bytes()...)
	logger.Info("[STEP] ingest complete", "total_buffer_bytes", len(fullBuffer))

	diarizeChan := make(chan siblingResult, 1)
	go func() {
		logger.Info("[STEP] call_diarize_on_end_byte: dispatching DiarizerMock")
		diarizeChan <- callHTTPSibling(ctx, "DiarizerMock", map[string]interface{}{
			"audio_b64": base64.StdEncoding.EncodeToString(fullBuffer),
		})
	}()

	transcribeWG.Wait()
	logger.Info("[STEP] all transcribe goroutines settled", "count", len(transcribeRes))
	diarize := <-diarizeChan

	sentences := extractSentences(transcribeRes)
	logger.Info("[STEP] call_assignment: dispatching AssignerMock", "sentences", len(sentences))
	assigner := callHTTPSibling(ctx, "AssignerMock", map[string]interface{}{
		"sentences": sentences,
		"diarize":   diarize.Body,
	})

	out := map[string]interface{}{
		"chainlet":         "EntryStreamingGo",
		"transcribe_calls": transcribeRes,
		"diarize":          diarize,
		"assigner":         assigner,
	}
	payload, _ := json.Marshal(out)
	logger.Info("[STEP] sending final reply", "bytes", len(payload))
	_ = conn.WriteMessage(websocket.TextMessage, payload)
	_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(1000, "done"), time.Now().Add(time.Second))
}

func extractSentences(rs []siblingResult) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(rs))
	for _, r := range rs {
		if r.Error != "" {
			continue
		}
		m, ok := r.Body.(map[string]interface{})
		if !ok {
			continue
		}
		out = append(out, m)
	}
	return out
}

func main() {
	logger = slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	if err := loadChainCfg(); err != nil {
		logger.Error("[INIT] loadChainCfg", "err", err)
		os.Exit(1)
	}
	if err := loadAPIKey(); err != nil {
		logger.Error("[INIT] loadAPIKey", "err", err)
		os.Exit(1)
	}
	logger.Info("[INIT] EntryStreamingGo ready", "siblings", len(chainCfg))
	for name, c := range chainCfg {
		logger.Info("[INIT] sibling", "name", name, "predict_url", c.PredictURL)
	}

	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/stream/v1", handleStream)
	logger.Info("[INIT] listening", "port", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		logger.Error("[INIT] listen failed", "err", err)
		os.Exit(1)
	}
}
