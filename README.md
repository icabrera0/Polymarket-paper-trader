# Polymarket Paper Trading Bot

Bot automatizado de **paper trading** para Polymarket que analiza noticias en tiempo real, detecta discrepancias entre el precio del mercado y la probabilidad inferida de los eventos, y simula operaciones aplicando reglas estrictas de gestión de riesgo. Genera un reporte Excel diario con el detalle de cada trade.

> ⚠️ **Fase 1: solo paper trading.** Ninguna operación real se ejecuta en Polymarket. Todo es simulación con balance virtual.

---

## Arquitectura

El bot se compone de 10 módulos orquestados en pipeline:

1. **NEWS_INGESTOR** — Ingesta de noticias (NewsAPI + GDELT), deduplicación y cola de prioridad.
2. **SENTIMENT_ANALYZER** — Análisis de sentimiento e impacto vía Claude API.
3. **MARKET_SCANNER** — Polling de la Gamma API de Polymarket; filtrado por volumen, spread y tiempo restante.
4. **DECISION_ENGINE** — Cruce de noticias y mercados; decisión de entrada basada en reglas + LLM.
5. **PAPER_TRADER** — Simulador de ejecución de órdenes con slippage y fills parciales.
6. **RISK_MANAGER** — Validación de cada trade contra los límites configurados (transversal).
7. **REPORT_GENERATOR** — Reporte Excel diario con 5 hojas (resumen, trades, noticias, métricas, evolución).
8. **NOTIFICATION_SYSTEM** — Alertas Discord en cada trade, stop loss, drawdown y resumen diario.
9. **ORCHESTRATOR** — `main.py` que coordina todos los módulos y planifica las tareas.
10. **BACKTESTING** — Replay histórico de noticias y mercados para validar la estrategia.

### Flujo de datos

```
News  →  Ingestor  →  Analyzer  →  Scanner  →  Decision Engine
                                                       │
                                                       ▼
Excel  ◄──  Report  ◄──  SQLite  ◄──  Paper Trader  ◄  Risk Manager
                                                       │
                                                       ▼
                                                    Discord
```

---

## Estructura del proyecto

```
Polymarket/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.yaml          # Toda la configuración tuneable
├── src/                       # Módulos (se crearán en partes posteriores)
├── data/                      # SQLite + caché de noticias
├── reports/                   # Excels diarios (YYYY-MM-DD_report.xlsx)
├── logs/                      # Logs de cada decisión del LLM
└── tests/                     # Tests unitarios
```

---

## Configuración inicial

### 1. Requisitos

- **Python 3.10+** (necesario para algunas anotaciones de tipos modernas)
- Cuenta en [Anthropic Console](https://console.anthropic.com) con API key
- Cuenta gratuita en [NewsAPI.org](https://newsapi.org/register)
- Webhook de Discord en el servidor donde quieras las notificaciones
- (GDELT no requiere API key)

### 2. Instalación

```bash
cd E:\AI\Polymarket
python -m venv venv
venv\Scripts\activate           # En Windows
pip install -r requirements.txt
```

### 3. Variables de entorno

Copia `.env.example` a `.env` y rellena con tus claves:

```bash
copy .env.example .env
```

Edita `.env`:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
NEWSAPI_KEY=xxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/xxxxx
```

### 4. Ajustes finos

Revisa `config/settings.yaml` para tunear parámetros sin tocar código (balance, % de riesgo, intervalos, etc.).

---

## Reglas de gestión de riesgo

Calibradas para un bankroll inicial de **150 €**:

| Parámetro | Valor | Notas |
|---|---|---|
| Balance inicial | 150 € | Configurable en `settings.yaml` |
| Tamaño máximo por posición | 15% del balance | ~22 € |
| Posiciones simultáneas máx. | 3 | |
| Tamaño mínimo de trade | 5 € | Evita operaciones irrelevantes |
| Stop loss | -20% del valor de entrada | Cierre automático |
| Take profit | +30% si la confianza disminuye | Cierre evaluado |
| Drawdown máximo | 30% del bankroll (45 €) | El bot se pausa automáticamente |
| Volumen 24h mínimo del mercado | $10,000 | Liquidez razonable |
| Spread máximo | 5 centavos | Evita mercados ineficientes |
| Edge mínimo precio vs probabilidad | 10% | Para entrar |

---

## Ejecución (disponible tras completar todos los módulos)

```bash
python -m src.main
```

El bot:
- Escanea Polymarket cada 5 minutos
- Sondea NewsAPI cada 5 minutos y GDELT cada 15
- Genera el reporte Excel diario a las 23:55 (hora Madrid)
- Envía notificaciones Discord en cada evento relevante

---

## Plan de desarrollo

Construcción incremental, módulo a módulo:

- [x] **Parte 1** — Estructura base + configuración
- [ ] **Parte 2** — RISK_MANAGER
- [ ] **Parte 3** — MARKET_SCANNER
- [ ] **Parte 4** — NEWS_INGESTOR
- [ ] **Parte 5** — SENTIMENT_ANALYZER
- [ ] **Parte 6** — DECISION_ENGINE
- [ ] **Parte 7** — PAPER_TRADER + SQLite
- [ ] **Parte 8** — REPORT_GENERATOR (Excel)
- [ ] **Parte 9** — NOTIFICATION_SYSTEM (Discord) + ORCHESTRATOR
- [ ] **Parte 10** — BACKTESTING

---

## Disclaimer

Este software es exclusivamente con **fines educativos y de investigación**. El paper trading no garantiza rendimientos en operaciones reales. Polymarket puede tener restricciones legales en tu jurisdicción — verifica antes de cualquier uso real. El autor no se responsabiliza de pérdidas derivadas del uso de este código.
