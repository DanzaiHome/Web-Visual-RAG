import React, { ChangeEvent, FormEvent, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ImagePlus, Loader2, MessageSquareText, Send, Settings2, X } from 'lucide-react';
import './styles.css';

type PreviewImage = {
  file: File;
  url: string;
};

type AskResponse = {
  answer: string;
  elapsed_seconds: number;
  image_count: number;
};

type AdvancedSettings = {
  topK: number;
  candidateK: number;
  chunkSize: number;
  chunksPerDoc: number;
  maxSufficiencyIterations: number;
  useMultimodal: boolean;
};

const initialSettings: AdvancedSettings = {
  topK: 5,
  candidateK: 10,
  chunkSize: 400,
  chunksPerDoc: 3,
  maxSufficiencyIterations: 3,
  useMultimodal: true,
};

function App() {
  const [question, setQuestion] = useState('');
  const [images, setImages] = useState<PreviewImage[]>([]);
  const [settings, setSettings] = useState<AdvancedSettings>(initialSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [answer, setAnswer] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const canSubmit = useMemo(() => question.trim().length > 0 && images.length > 0 && !isLoading, [question, images, isLoading]);

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    const nextImages = files.slice(0, 4).map((file) => ({ file, url: URL.createObjectURL(file) }));
    images.forEach((image) => URL.revokeObjectURL(image.url));
    setImages(nextImages);
    setError('');
  }

  function removeImage(index: number) {
    setImages((current) => {
      const target = current[index];
      if (target) URL.revokeObjectURL(target.url);
      return current.filter((_, itemIndex) => itemIndex !== index);
    });
  }

  function updateSetting<K extends keyof AdvancedSettings>(key: K, value: AdvancedSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) return;

    const payload = new FormData();
    payload.append('question', question.trim());
    images.forEach((image) => payload.append('images', image.file));
    payload.append('top_k', String(settings.topK));
    payload.append('candidate_k', String(settings.candidateK));
    payload.append('chunk_size', String(settings.chunkSize));
    payload.append('chunks_per_doc', String(settings.chunksPerDoc));
    payload.append('max_sufficiency_iterations', String(settings.maxSufficiencyIterations));
    payload.append('use_multimodal', String(settings.useMultimodal));

    setIsLoading(true);
    setError('');
    setAnswer('');
    setElapsedSeconds(null);

    try {
      const response = await fetch(new URL('api/ask', window.location.href), {
        method: 'POST',
        body: payload,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.detail || `Request failed with status ${response.status}`);
      }
      const result = data as AskResponse;
      setAnswer(result.answer);
      setElapsedSeconds(result.elapsed_seconds);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '请求失败，请检查后端服务和 API Key 配置。');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Visual Web RAG</h1>
            <p>上传图片并提问，系统会检索网页证据后生成带引用的回答。</p>
          </div>
          <button className="iconButton" type="button" onClick={() => setShowSettings((value) => !value)} aria-label="检索参数">
            <Settings2 size={20} />
          </button>
        </header>

        <form className="panel inputPanel" onSubmit={handleSubmit}>
          <label className="dropzone">
            <input type="file" accept="image/*" multiple onChange={handleImageChange} />
            <ImagePlus size={28} />
            <span>选择 1-4 张图片</span>
          </label>

          {images.length > 0 && (
            <div className="previewGrid">
              {images.map((image, index) => (
                <figure className="preview" key={image.url}>
                  <img src={image.url} alt={`上传图片 ${index + 1}`} />
                  <button type="button" onClick={() => removeImage(index)} aria-label="移除图片">
                    <X size={15} />
                  </button>
                </figure>
              ))}
            </div>
          )}

          <label className="questionBox">
            <span>问题</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="例如：这张图中的人物最近宣布了哪些政策？请给出证据。"
              rows={5}
            />
          </label>

          {showSettings && (
            <div className="settingsGrid">
              <NumberField label="Top K" value={settings.topK} min={1} max={12} onChange={(value) => updateSetting('topK', value)} />
              <NumberField label="候选网页" value={settings.candidateK} min={1} max={30} onChange={(value) => updateSetting('candidateK', value)} />
              <NumberField label="Chunk 大小" value={settings.chunkSize} min={120} max={1200} step={20} onChange={(value) => updateSetting('chunkSize', value)} />
              <NumberField label="每文档 Chunk" value={settings.chunksPerDoc} min={1} max={8} onChange={(value) => updateSetting('chunksPerDoc', value)} />
              <NumberField label="补充检索轮数" value={settings.maxSufficiencyIterations} min={0} max={5} onChange={(value) => updateSetting('maxSufficiencyIterations', value)} />
              <label className="toggleField">
                <input
                  type="checkbox"
                  checked={settings.useMultimodal}
                  onChange={(event) => updateSetting('useMultimodal', event.target.checked)}
                />
                <span>启用多模态召回</span>
              </label>
            </div>
          )}

          <button className="submitButton" type="submit" disabled={!canSubmit}>
            {isLoading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            {isLoading ? '正在检索与生成' : '提交问题'}
          </button>
        </form>
      </section>

      <section className="panel answerPanel">
        <div className="answerHeader">
          <div>
            <h2>模型回复</h2>
            {elapsedSeconds !== null && <span>{elapsedSeconds.toFixed(1)} 秒</span>}
          </div>
          <MessageSquareText size={22} />
        </div>
        {error && <div className="errorBox">{error}</div>}
        {!error && !answer && !isLoading && <div className="emptyState">回答会显示在这里。</div>}
        {isLoading && <div className="emptyState">正在调用 Visual Web RAG pipeline，请等待检索、网页抓取和生成完成。</div>}
        {answer && <pre className="answerText">{answer}</pre>}
      </section>
    </main>
  );
}

function NumberField({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return (
    <label className="numberField">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
