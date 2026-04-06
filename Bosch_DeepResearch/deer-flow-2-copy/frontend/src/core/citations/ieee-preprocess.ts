/**
 * CitationMiddleware outputs IEEE-style body text ("claim[1][2].") plus a
 * "## 参考文献" / "## References" block with rows `[n] "Title". https://…`.
 * The UI only treats markdown links `[citation:…](url)` as CitationLink chips,
 * so we rewrite numeric cites in the body using the parsed bibliography map.
 */

const REF_HEADING_RE =
  /(?:^|\n)\s*(?:#{1,2}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?(?:References|参考文献|Bibliography|参考资料)\s*(?:$|\n)/i;

/** Same order of magnitude as backend `\[[1-9]\d{0,2}\]` (1–999). */
const BODY_NUMERIC_CITE_RE = /\[[1-9]\d{0,2}\]/g;

/** `[n] "title". https://url` possibly after list marker / whitespace. */
const REF_ENTRY_RE =
  /(?:^|\n)\s*(?:[-*+]\s*)?\[(\d+)\]\s*"((?:[^"\\]|\\.)*)"\.\s+(https?:\/\/\S+)/g;

function escapeMarkdownLinkLabel(text: string): string {
  return text.replace(/\[/g, "‹").replace(/\]/g, "›").replace(/\(/g, "⦅").replace(/\)/g, "⦆");
}

export function parseIeeeReferenceMap(refSectionMarkdown: string): Map<
  number,
  { title: string; url: string }
> {
  const map = new Map<number, { title: string; url: string }>();
  let m: RegExpExecArray | null;
  const re = new RegExp(REF_ENTRY_RE.source, "g");
  while ((m = re.exec(refSectionMarkdown)) !== null) {
    const idx = m[1];
    const rawTitle = m[2];
    const rawUrl = m[3];
    if (idx === undefined || rawTitle === undefined || rawUrl === undefined) {
      continue;
    }
    const n = Number.parseInt(idx, 10);
    const title = rawTitle.replace(/\\"/g, '"').trim();
    const url = rawUrl.trim();
    if (!Number.isFinite(n) || n < 1) continue;
    map.set(n, { title, url });
  }
  return map;
}

function injectNumericCitations(
  body: string,
  map: Map<number, { title: string; url: string }>,
): string {
  if (map.size === 0) return body;
  return body.replace(BODY_NUMERIC_CITE_RE, (full) => {
    const n = Number.parseInt(full.slice(1, -1), 10);
    const ref = map.get(n);
    if (!ref) return full;
    const rawLabel =
      ref.title.trim() || `Source ${n}`;
    const label = escapeMarkdownLinkLabel(rawLabel);
    return `[citation:${label}](${ref.url})`;
  });
}

/**
 * If the markdown contains an IEEE bibliography section, replace body `[n]`
 * markers with `[citation:title](url)` so CitationLink renders. Leaves
 * content unchanged when no reference block or no parseable entries.
 */
export function preprocessIeeeNumericCitations(markdown: string): string {
  const text = markdown ?? "";
  if (!text.includes("[") || !/\[[1-9]\d{0,2}\]/.test(text)) {
    return text;
  }

  const headingMatch = REF_HEADING_RE.exec(text);
  if (!headingMatch) {
    return text;
  }

  const body = text.slice(0, headingMatch.index);
  const fromHeading = text.slice(headingMatch.index);
  const map = parseIeeeReferenceMap(fromHeading);
  if (map.size === 0) {
    return text;
  }

  const newBody = injectNumericCitations(body, map);
  return newBody + fromHeading;
}
