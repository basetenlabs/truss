package main

import (
	"encoding/json"
	"net/http"
	"os"
)

type HealthResponse struct {
	Message string `json:"message"`
}

type PredictResponse struct {
	Message         string `json:"message"`
	IsEnvVarPassed  bool   `json:"is_env_var_passed"`
	IsSecretMounted bool   `json:"is_secret_mounted"`
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	response := HealthResponse{Message: "OK"}
	json.NewEncoder(w).Encode(response)
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Check if HF_TOKEN environment variable exists
	_, envExists := os.LookupEnv("HF_TOKEN")

	// Check if secret file exists
	_, err := os.Stat("/secrets/hf_access_token")
	secretExists := err == nil

	response := PredictResponse{
		Message:         "Hello World",
		IsEnvVarPassed:  envExists,
		IsSecretMounted: secretExists,
	}
	json.NewEncoder(w).Encode(response)
}

func main() {
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/predict", predictHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	println("Server starting on port", port)
	http.ListenAndServe(":"+port, nil)
}
