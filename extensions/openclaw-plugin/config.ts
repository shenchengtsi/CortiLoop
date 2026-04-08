export type CortiLoopPluginConfig = {
  cortiloopUrl: string;
  autoCapture: boolean;
  autoRecall: boolean;
  recallTopK: number;
  captureMaxChars: number;
};

const DEFAULT_URL = "http://127.0.0.1:8766";
const DEFAULT_RECALL_TOP_K = 5;
const DEFAULT_CAPTURE_MAX_CHARS = 2000;

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length > 0) {
    throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
  }
}

export const cortiloopConfigSchema = {
  parse(value: unknown): CortiLoopPluginConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      // Use all defaults if no config provided
      return {
        cortiloopUrl: DEFAULT_URL,
        autoCapture: true,
        autoRecall: true,
        recallTopK: DEFAULT_RECALL_TOP_K,
        captureMaxChars: DEFAULT_CAPTURE_MAX_CHARS,
      };
    }

    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["cortiloopUrl", "autoCapture", "autoRecall", "recallTopK", "captureMaxChars"],
      "cortiloop config",
    );

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : DEFAULT_CAPTURE_MAX_CHARS;
    if (captureMaxChars < 100 || captureMaxChars > 10_000) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    return {
      cortiloopUrl: typeof cfg.cortiloopUrl === "string" ? cfg.cortiloopUrl : DEFAULT_URL,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
      recallTopK: typeof cfg.recallTopK === "number" ? cfg.recallTopK : DEFAULT_RECALL_TOP_K,
      captureMaxChars,
    };
  },
};
