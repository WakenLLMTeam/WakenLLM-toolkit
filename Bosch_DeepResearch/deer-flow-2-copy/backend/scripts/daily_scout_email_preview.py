import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.daily_scout_email_template import (
    build_bosch_email_html,
    build_inline_asset_sources,
    load_bosch_inline_assets,
    render_markdown_html,
)
from scripts.daily_scout_accounts import DEFAULT_SCOUT_TIMEZONE


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Bosch-style Daily Scout HTML preview without running report generation")
    parser.add_argument("report", type=Path, help="Path to the markdown report file")
    parser.add_argument("--output", type=Path, help="Path to the generated preview HTML file")
    parser.add_argument("--topic", default="Daily Scout Report", help="Title shown in the preview header")
    parser.add_argument("--recipient", default="preview@example.com", help="Recipient label shown in the preview")
    parser.add_argument("--trigger", default="Manual Preview", help="Trigger label shown in the preview")
    parser.add_argument("--report-name", help="Optional report name shown below the title")
    parser.add_argument("--generated-at", help="Optional timestamp label; defaults to current Shanghai time")
    parser.add_argument("--template-dir", type=Path, help="Optional override directory containing headerbar.png and boschLogo.png")
    args = parser.parse_args()

    report_path = args.report.expanduser().resolve()
    report_text = report_path.read_text(encoding="utf-8")
    report_html = render_markdown_html(report_text)
    assets = load_bosch_inline_assets(args.template_dir)
    asset_sources = build_inline_asset_sources(assets, mode="data-uri")
    generated_at = args.generated_at or datetime.now(ZoneInfo(DEFAULT_SCOUT_TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")
    output_path = (args.output or report_path.with_suffix(".preview.html")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_text = build_bosch_email_html(
        title=args.topic,
        recipient_email=args.recipient,
        trigger_label=args.trigger,
        generated_at=generated_at,
        report_html=report_html,
        report_name=args.report_name or report_path.name,
        asset_sources=asset_sources,
    )
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())