import base64
import html
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("daily_scout_email_template")

_EMAIL_TEMPLATE_DIR_ENV = "DAILY_SCOUT_EMAIL_TEMPLATE_DIR"
_DEFAULT_TEMPLATE_DIR_NAME = "email_assets"


@dataclass(frozen=True)
class InlineImageAsset:
    key: str
    filename: str
    content_id: str
    mime_subtype: str
    data: bytes


def _default_template_dir() -> Path:
  return Path(__file__).resolve().parent / _DEFAULT_TEMPLATE_DIR_NAME


def resolve_email_template_dir(template_dir: str | Path | None = None) -> Path | None:
    raw_value = template_dir or os.getenv(_EMAIL_TEMPLATE_DIR_ENV, "").strip()
    candidate = Path(raw_value).expanduser() if raw_value else _default_template_dir()
    if candidate.exists() and candidate.is_dir():
        return candidate
    logger.warning("Daily scout email template directory %s is unavailable; falling back to no inline branding", candidate)
    return None


def load_bosch_inline_assets(template_dir: str | Path | None = None) -> dict[str, InlineImageAsset]:
    resolved_dir = resolve_email_template_dir(template_dir)
    if resolved_dir is None:
        return {}

    asset_specs = {
        "headerbar": ("headerbar.png", "daily-scout-headerbar"),
        "logo": ("boschLogo.png", "daily-scout-bosch-logo"),
    }
    assets: dict[str, InlineImageAsset] = {}
    for key, (filename, content_id) in asset_specs.items():
        path = resolved_dir / filename
        if not path.exists() or not path.is_file():
            logger.warning("Daily scout email asset %s is missing", path)
            continue
        try:
            assets[key] = InlineImageAsset(
                key=key,
                filename=filename,
                content_id=content_id,
                mime_subtype=path.suffix.lstrip(".").lower() or "png",
                data=path.read_bytes(),
            )
        except Exception:
            logger.exception("Failed to read daily scout email asset %s", path)
    return assets


def build_inline_asset_sources(assets: dict[str, InlineImageAsset], *, mode: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for key, asset in assets.items():
        if mode == "cid":
            sources[key] = f"cid:{asset.content_id}"
            continue
        if mode == "data-uri":
            encoded = base64.b64encode(asset.data).decode("ascii")
            sources[key] = f"data:image/{asset.mime_subtype};base64,{encoded}"
            continue
        raise ValueError(f"Unsupported inline asset mode: {mode}")
    return sources


def render_markdown_html(markdown_text: str) -> str:
    try:
        markdown_module = importlib.import_module("markdown")
        rendered = markdown_module.markdown(
            markdown_text,
            extensions=["extra", "sane_lists", "nl2br", "toc"],
        )
        if rendered.strip():
            return rendered
    except Exception:
        logger.exception("Failed to render markdown report for email; falling back to preformatted HTML")

    return f"<pre>{html.escape(markdown_text)}</pre>"


def build_bosch_email_html(
    *,
    title: str,
    recipient_email: str,
    trigger_label: str,
    generated_at: str,
    report_html: str,
    report_name: str,
    asset_sources: dict[str, str] | None = None,
) -> str:
    asset_sources = asset_sources or {}
    headerbar_src = asset_sources.get("headerbar", "")
    logo_src = asset_sources.get("logo", "")
    headerbar_html = (
        f'<img src="{headerbar_src}" alt="Bosch report bar" style="display:block;width:100%;height:auto;border:0;" />'
        if headerbar_src
        else '<div style="height:18px;background:#0b6fd3;border-radius:2px;"></div>'
    )
    logo_html = (
        f'<img src="{logo_src}" alt="Bosch logo" style="display:block;height:40px;width:auto;border:0;" />'
        if logo_src
        else '<div style="font:700 22px/1.2 Arial,sans-serif;color:#0b6fd3;">Bosch</div>'
    )
    footer_logo_html = (
        f'<img src="{logo_src}" alt="Bosch logo" style="display:inline-block;height:48px;width:auto;border:0;opacity:0.72;" />'
        if logo_src
        else '<div style="font:700 22px/1.2 Arial,sans-serif;color:#0b6fd3;opacity:0.72;">Bosch</div>'
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; padding: 0; background: #f0f0f0; color: #333333; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif; }}
    .background-block {{ background: #f0f0f0; padding: 32px 10px; }}
    .content-block {{ max-width: 800px; margin: 0 auto; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); overflow: hidden; }}
    .inner {{ padding: 30px; }}
    .header-table, .meta-table {{ width: 100%; border-collapse: collapse; }}
    .report-date {{ color: #888888; font-size: 14px; text-align: right; }}
    .main-title {{ margin: 20px 0 20px; color: #4a90e2; font-size: 28px; line-height: 1.2; font-weight: 700; text-align: center; }}
    .meta-table td {{ width: 33.33%; padding: 0 6px 12px; vertical-align: top; }}
    .meta-card {{ background: #f7f9fc; border: 1px solid #dfe8f5; border-radius: 6px; padding: 14px 16px; }}
    .meta-label {{ margin: 0 0 6px; color: #4a90e2; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
    .meta-value {{ margin: 0; color: #333333; font-size: 14px; line-height: 1.5; word-break: break-word; }}
    .subtitle-bar {{ display: inline-block; margin: 28px 0 18px; background: #4a90e2; color: #ffffff; padding: 12px 20px; border-radius: 5px; font-size: 20px; font-weight: 700; min-width: 240px; }}
    .report-body {{ color: #555555; font-size: 16px; line-height: 1.7; }}
    .report-body > :first-child {{ margin-top: 0; }}
    .report-body h1, .report-body h2, .report-body h3, .report-body h4 {{ color: #333333; line-height: 1.25; margin: 1.6em 0 0.55em; }}
    .report-body h1 {{ font-size: 24px; }}
    .report-body h2 {{ font-size: 20px; border-bottom: 1px solid #eeeeee; padding-bottom: 8px; }}
    .report-body h3 {{ font-size: 18px; }}
    .report-body p, .report-body ul, .report-body ol, .report-body blockquote, .report-body pre, .report-body table {{ margin: 0 0 1em; }}
    .report-body ul, .report-body ol {{ padding-left: 22px; }}
    .report-body li {{ margin-bottom: 0.35em; }}
    .report-body a {{ color: #4a90e2; text-decoration: none; font-weight: 600; }}
    .report-body strong {{ color: #333333; }}
    .report-body code {{ padding: 2px 5px; border-radius: 4px; background: #eef5fd; color: #175a9d; font-size: 0.92em; }}
    .report-body pre {{ padding: 16px; border-radius: 6px; background: #f7f9fc; overflow-x: auto; }}
    .report-body pre code {{ padding: 0; background: transparent; color: inherit; }}
    .report-body blockquote {{ margin-left: 0; padding: 12px 16px; border-left: 4px solid #4a90e2; background: #f7f9fc; color: #4b5563; }}
    .report-body table {{ width: 100%; border-collapse: collapse; }}
    .report-body th, .report-body td {{ border: 1px solid #dfe3ea; padding: 10px 12px; text-align: left; vertical-align: top; }}
    .report-body th {{ background: #f3f7fc; color: #333333; }}
    .footer-note {{ margin: 24px 0 0; color: #888888; font-size: 13px; line-height: 1.6; text-align: center; }}
    .footer-logo {{ padding: 20px 30px 20px; text-align: center; }}
    @media (max-width: 640px) {{
      .background-block {{ padding: 16px 8px; }}
      .inner {{ padding: 22px 18px; }}
      .header-table, .header-table tbody, .header-table tr, .header-table td, .meta-table, .meta-table tbody, .meta-table tr, .meta-table td {{ display: block; width: 100% !important; }}
      .report-date {{ text-align: left; padding-top: 16px; }}
      .meta-table td {{ padding: 0 0 10px; }}
      .main-title {{ font-size: 24px; }}
      .subtitle-bar {{ width: auto; min-width: 0; display: block; text-align: center; }}
    }}
  </style>
</head>
<body>
  <div class=\"background-block\">
    <table role=\"presentation\" class=\"content-block\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\">
      <tr>
        <td>{headerbar_html}</td>
      </tr>
      <tr>
        <td class=\"inner\">
          <table role=\"presentation\" class=\"header-table\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\">
            <tr>
              <td style=\"vertical-align:middle;\">{logo_html}</td>
              <td class=\"report-date\" style=\"vertical-align:middle;\">{html.escape(generated_at)}</td>
            </tr>
          </table>
          <h1 class=\"main-title\">Daily News Scouting</h1>
          <table role=\"presentation\" class=\"meta-table\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\">
            <tr>
              <td>
                <div class=\"meta-card\">
                  <p class=\"meta-label\">Recipient</p>
                  <p class=\"meta-value\">{html.escape(recipient_email)}</p>
                </div>
              </td>
              <td>
                <div class=\"meta-card\">
                  <p class=\"meta-label\">Trigger</p>
                  <p class=\"meta-value\">{html.escape(trigger_label)}</p>
                </div>
              </td>
              <td>
                <div class=\"meta-card\">
                  <p class=\"meta-label\">Generated</p>
                  <p class=\"meta-value\">{html.escape(generated_at)}</p>
                </div>
              </td>
            </tr>
          </table>
          <div class=\"subtitle-bar\">Daily Brief</div>
          <div class=\"report-body\">{report_html}</div>
          <p class=\"footer-note\">Generated by XC Daily Scout Agent.</p>
        </td>
      </tr>
      <tr>
        <td>{headerbar_html}</td>
      </tr>
      <tr>
        <td class=\"footer-logo\">{footer_logo_html}</td>
      </tr>
    </table>
  </div>
</body>
</html>"""