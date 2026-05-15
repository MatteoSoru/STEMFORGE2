# StemForge — GitHub Actions DMG Build

Questi file generano automaticamente `StemForge-1.0.0-macOS.dmg` su un runner macOS di GitHub, senza che tu debba compilare nulla sul tuo Mac.

---

## Setup in 5 passi

### 1. Crea un repository GitHub

Vai su [github.com/new](https://github.com/new), crea un repo (anche privato va bene).

### 2. Carica tutti i file del progetto

Struttura attesa nel repo:

```
stemforge/
├── .github/
│   └── workflows/
│       └── build.yml          ← workflow Actions
├── scripts/
│   ├── make_icon.py
│   └── make_dmg_background.py
├── launcher.py
├── stemforge.spec
├── api_server_v2.py            ← dai file della conversazione precedente
├── multi_generator.py
├── stem_splitter.py
├── music_generator.py
├── setup_models.py
└── requirements.txt
```

Puoi caricarli con GitHub Desktop oppure da terminale:

```bash
git init
git add .
git commit -m "Initial StemForge commit"
git remote add origin https://github.com/TUO_USERNAME/stemforge.git
git push -u origin main
```

### 3. La build parte automaticamente

Ogni push su `main` avvia il workflow. Puoi vederlo in:
**Repository → Actions → Build StemForge DMG**

La build impiega circa 15-25 minuti (scarica PyTorch, AudioCraft, ecc.).

### 4. Scarica il DMG

Al termine della build:
- **Actions → ultima run → Artifacts → StemForge-1.0.0-macOS**
- Clicca per scaricare lo zip che contiene il DMG

### 5. (Opzionale) Crea una Release con il DMG allegato

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub creerà automaticamente una Release con il DMG allegato per il download diretto.

---

## Note importanti

| Cosa | Dettaglio |
|------|-----------|
| Firma | Ad-hoc (no Developer ID) — al primo avvio: tasto destro → Apri → Apri comunque |
| Modelli AI | Scaricati al primo avvio (~3 GB), salvati in `~/.cache/stemforge/` |
| Apple Silicon | PyTorch usa MPS → generazione 3× più veloce |
| macOS richiesto | 13 Ventura o superiore |

---

## Avvio manuale della build (senza push)

Dalla pagina Actions del repo:
**Build StemForge DMG → Run workflow → Run workflow**
