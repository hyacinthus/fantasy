package google

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// Thought signatures are opaque binary: NUL bytes and invalid UTF-8.
var rawSignature = []byte("\x12\x00\xff\xfe\x80sig\x00\xc3")

// TestThoughtSignatureSurvivesJSON streams a signed function call, stores
// the turn as JSON the way an app persists history, and checks the next
// request replays the exact signature.
func TestThoughtSignatureSurvivesJSON(t *testing.T) {
	wire := base64.StdEncoding.EncodeToString(rawSignature)
	// The handler only records: assertions run on the test goroutine.
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, body)

		part := map[string]any{"text": "done"}
		if len(bodies) == 1 {
			part = map[string]any{
				"functionCall":     map[string]any{"name": "echo", "args": map[string]any{}},
				"thoughtSignature": wire,
			}
		}
		chunk, _ := json.Marshal(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{part}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + string(chunk) + "\r\n\r\n"))
	}))
	defer server.Close()
	provider, err := New(WithVertex("test-project", "us-central1"), WithBaseURL(server.URL), WithSkipAuth(true))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gemini-3-flash")
	require.NoError(t, err)

	prompt := fantasy.Prompt{fantasy.NewUserMessage("Hi")}
	stream, err := model.Stream(t.Context(), fantasy.Call{Prompt: prompt})
	require.NoError(t, err)
	var metadata *ReasoningMetadata
	var call fantasy.StreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.StreamPartTypeReasoningEnd:
			metadata = GetReasoningMetadata(fantasy.ProviderOptions(part.ProviderMetadata))
		case fantasy.StreamPartTypeToolCall:
			call = part
		case fantasy.StreamPartTypeError:
			t.Fatal(part.Error)
		}
	}
	require.NotNil(t, metadata)
	require.Equal(t, wire, metadata.Signature, "stored in the wire's base64 form")

	prompt = append(prompt,
		fantasy.Message{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{Name: metadata}},
			fantasy.ToolCallPart{ToolCallID: call.ID, ToolName: call.ToolCallName, Input: call.ToolCallInput},
		}},
		fantasy.Message{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{ToolCallID: call.ID, Output: fantasy.ToolResultOutputContentText{Text: "ok"}},
		}},
	)
	saved, err := json.Marshal(prompt)
	require.NoError(t, err)
	var restored fantasy.Prompt
	require.NoError(t, json.Unmarshal(saved, &restored))

	stream, err = model.Stream(t.Context(), fantasy.Call{Prompt: restored})
	require.NoError(t, err)
	for range stream { //nolint:revive // drain
	}
	require.Len(t, bodies, 2)
	var replay struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	require.NoError(t, json.Unmarshal(bodies[1], &replay))
	require.Equal(t, wire, replay.Contents[1].Parts[0]["thoughtSignature"])
}

func TestDecodeSignatureKeepsLegacyRawValue(t *testing.T) {
	require.Equal(t, rawSignature, decodeSignature(base64.StdEncoding.EncodeToString(rawSignature)))
	// Older versions stored the raw bytes, which are not valid base64.
	require.Equal(t, rawSignature, decodeSignature(string(rawSignature)))
}
