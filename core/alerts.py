from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd


@dataclass
class SmtpConfig:
    host: str
    port: int
    user: str
    password: str
    from_addr: str
    to_addr: str


def build_alert_html(result_df: pd.DataFrame, threshold_pp: float, total_eur: float) -> str:
    deviating = result_df[result_df["Desvio Atual (pp)"].abs() >= threshold_pp]
    rows_html = ""
    for _, row in deviating.iterrows():
        color = "#c0392b" if row["Desvio Atual (pp)"] > 0 else "#2980b9"
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 12px;border:1px solid #ddd'>{row['Ticker']}</td>"
            f"<td style='padding:6px 12px;border:1px solid #ddd;text-align:right'>"
            f"{row['Percentagem Alvo (%)']:.2f}%</td>"
            f"<td style='padding:6px 12px;border:1px solid #ddd;text-align:right'>"
            f"{row['Alocação Atual (%)']:.2f}%</td>"
            f"<td style='padding:6px 12px;border:1px solid #ddd;text-align:right;"
            f"color:{color};font-weight:bold'>{row['Desvio Atual (pp)']:+.2f} pp</td>"
            f"</tr>"
        )
    from datetime import datetime
    return f"""
<html><body style='font-family:Arial,sans-serif;color:#222'>
<h2 style='color:#2c3e50'>&#128202; Alerta de Rebalanceamento</h2>
<p>Valor total do portfólio: <strong>€{total_eur:,.2f}</strong></p>
<p>Os seguintes ativos excedem o limiar de <strong>{threshold_pp:.1f} pp</strong>:</p>
<table style='border-collapse:collapse;font-size:14px'>
  <thead>
    <tr style='background:#ecf0f1'>
      <th style='padding:6px 12px;border:1px solid #ddd;text-align:left'>Ticker</th>
      <th style='padding:6px 12px;border:1px solid #ddd'>Alvo (%)</th>
      <th style='padding:6px 12px;border:1px solid #ddd'>Atual (%)</th>
      <th style='padding:6px 12px;border:1px solid #ddd'>Desvio (pp)</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style='color:#888;font-size:12px;margin-top:20px'>
  Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Otimizador de Portfólio v0.4.0
</p>
</body></html>"""


def send_alert_email(config: SmtpConfig, subject: str, html_body: str) -> tuple[bool, str]:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.from_addr
        msg["To"] = config.to_addr
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        ctx = ssl.create_default_context()
        if config.port == 465:
            with smtplib.SMTP_SSL(config.host, config.port, context=ctx) as srv:
                srv.login(config.user, config.password)
                srv.sendmail(config.from_addr, config.to_addr, msg.as_string())
        else:
            with smtplib.SMTP(config.host, config.port, timeout=15) as srv:
                srv.ehlo()
                srv.starttls(context=ctx)
                srv.login(config.user, config.password)
                srv.sendmail(config.from_addr, config.to_addr, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)
