/**
 * @license
 * Copyright 2026 Qwen
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { StreamingState } from '../types.js';

export interface LlamaCppProgress {
  /** Total prompt tokens to process. */
  total: number;
  /** Prompt tokens processed so far. */
  processed: number;
  /** Prompt tokens served from KV cache (instant). */
  cached: number;
  /** Progress fraction 0–1. */
  fraction: number;
  /** Estimated seconds remaining, or null if not enough data. */
  etaSeconds: number | null;
}

interface SlotData {
  is_processing: boolean;
  n_prompt_tokens: number;
  n_prompt_tokens_processed: number;
  n_prompt_tokens_cache: number;
}

/**
 * Polls the llama.cpp `/slots` endpoint during prompt processing to report
 * progress. Automatically disables itself on 404 or empty responses.
 *
 * @param slotsBaseUrl  Base URL of the llama.cpp server (e.g. "https://host:port")
 * @param modelId       Active model ID, appended as ?model=<id>
 * @param streamingState Current streaming state
 * @param isReceivingContent True once the first token arrives
 */
export function useLlamaCppProgress(
  slotsBaseUrl: string | undefined,
  modelId: string | undefined,
  streamingState: StreamingState,
  isReceivingContent: boolean,
): LlamaCppProgress | null {
  const [progress, setProgress] = useState<LlamaCppProgress | null>(null);
  // Track whether this endpoint is available. Once we get a 404 or error,
  // stop polling for the rest of this session.
  const disabledRef = useRef(false);
  // For ETA calculation: track (timestamp, tokensProcessed) samples.
  const samplesRef = useRef<Array<{ time: number; processed: number }>>([]);

  const isActive =
    !!slotsBaseUrl &&
    !!modelId &&
    streamingState === StreamingState.Responding &&
    !isReceivingContent &&
    !disabledRef.current;

  const poll = useCallback(async () => {
    if (!slotsBaseUrl || !modelId) return;

    const url = `${slotsBaseUrl.replace(/\/+$/, '')}/slots?model=${encodeURIComponent(modelId)}`;

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);

      if (!res.ok) {
        disabledRef.current = true;
        setProgress(null);
        return;
      }

      const data: SlotData[] = await res.json();
      if (!Array.isArray(data) || data.length === 0) {
        disabledRef.current = true;
        setProgress(null);
        return;
      }

      // Find an actively processing slot.
      const active = data.find((s) => s.is_processing && s.n_prompt_tokens > 0);
      if (!active) {
        // Server is fine but no active slot — just no progress to show yet.
        return;
      }

      const total = active.n_prompt_tokens;
      const processed = active.n_prompt_tokens_processed;
      const cached = active.n_prompt_tokens_cache;
      const fraction = total > 0 ? Math.min(1, processed / total) : 0;

      // ETA calculation based on processing rate.
      const now = Date.now();
      const samples = samplesRef.current;
      samples.push({ time: now, processed });
      // Keep only samples from the last 30 seconds.
      const cutoff = now - 30_000;
      while (samples.length > 0 && samples[0]!.time < cutoff) {
        samples.shift();
      }

      let etaSeconds: number | null = null;
      if (samples.length >= 2) {
        const first = samples[0]!;
        const last = samples[samples.length - 1]!;
        const dt = (last.time - first.time) / 1000; // seconds
        const dp = last.processed - first.processed; // tokens
        if (dt > 0 && dp > 0) {
          const rate = dp / dt; // tokens per second
          const remaining = total - processed;
          etaSeconds = Math.max(0, Math.round(remaining / rate));
        }
      }

      setProgress({ total, processed, cached, fraction, etaSeconds });
    } catch {
      // Network error or abort — disable to avoid spamming.
      disabledRef.current = true;
      setProgress(null);
    }
  }, [slotsBaseUrl, modelId]);

  useEffect(() => {
    if (!isActive) {
      // Clear progress and samples when we leave the active state.
      setProgress(null);
      samplesRef.current = [];
      return;
    }

    // Poll immediately then every 1 second.
    poll();
    const interval = setInterval(poll, 1000);
    return () => clearInterval(interval);
  }, [isActive, poll]);

  // Reset disabled flag when model changes (different server config maybe).
  useEffect(() => {
    disabledRef.current = false;
  }, [slotsBaseUrl, modelId]);

  return isActive ? progress : null;
}

/**
 * Format prompt progress for display.
 * Example: "Processing prompt 40960/49152 (83%) ~2m45s remaining"
 */
export function formatPromptProgress(p: LlamaCppProgress): string {
  const pct = Math.round(p.fraction * 100);
  const processedStr = formatTokenCount(p.processed);
  const totalStr = formatTokenCount(p.total);
  let text = `Processing prompt ${processedStr}/${totalStr} (${pct}%)`;

  if (p.etaSeconds !== null && p.etaSeconds > 0) {
    text += ` ~${formatEta(p.etaSeconds)} remaining`;
  }

  return text;
}

function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}m`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m${s.toString().padStart(2, '0')}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h${rm.toString().padStart(2, '0')}m`;
}
