# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` in the repo root is a symlink to this file (`AGENTS.md -> CLAUDE.md`). All AI coding agents (Claude Code, Codex, Cursor, etc.) read the same instructions — edit `CLAUDE.md` only; do not replace the symlink with a separate file.

## Working with `gh` (GitHub CLI)

Before running ANY `gh` command (creating PRs/issues, commenting, merging, releasing, etc.), you MUST:

1. Run `git config user.name` and `git config user.email` to identify the current repo's git author.
2. Run `gh auth status` to list every GitHub account `gh` knows about (active and inactive). Identify the account whose login (or associated email) matches the git user from step 1 — call this `<MATCHED_USER>`.
3. Run every `gh` invocation with that account's token injected via env var, NOT by switching the active account. The verified pattern is:

   ```bash
   GH_TOKEN=$(gh auth token --user <MATCHED_USER>) gh <subcommand> ...
   ```

   `GH_TOKEN` takes precedence over `gh`'s stored active account for that single process, so the command runs as `<MATCHED_USER>` regardless of which account is "active". Verify with `GH_TOKEN=$(gh auth token --user <MATCHED_USER>) gh api user --jq .login` — it must print `<MATCHED_USER>`.
4. If no matching account can be found in `gh auth status` (no auth, no account matches the git user, ambiguous mapping), **refuse to run the `gh` command** and report the mismatch to the user. Do NOT guess, do NOT proceed with a non-matching account, and do NOT silently use whatever account `gh` defaults to.

### Forbidden: `gh auth switch`

Do **NOT** use `gh auth switch` to change accounts before running `gh` commands. `gh auth switch` mutates global state in `~/.config/gh/hosts.yml` (the "active account" pointer); when multiple processes/agents run concurrently on this machine, one process's switch silently changes the active account under another process's feet, causing PRs/comments to be created under the wrong identity. Always use the per-process `GH_TOKEN=$(gh auth token --user ...)` pattern instead — it scopes the account choice to one command and cannot race with other processes.

Rationale: this repo has been touched by multiple git identities, and this machine routinely runs concurrent agents/automation. Running `gh` under the wrong account creates PRs/comments attributed to the wrong person and is hard to undo.

## Project Overview

This is a web scraping and data archival repository that crawls artwork URLs from Skeb.jp posts and downloads content from various file hosting services (Google Drive, Imgur, Dropbox) to HuggingFace datasets. The `pyskeb` package is a minimal client wrapper; the core functionality is in `test/prepare/` scripts.

## Environment Setup

Required environment variables:
- `HF_TOKEN`: HuggingFace API token for dataset uploads
- `REMOTE_REPOSITORY`: Target HuggingFace dataset repository ID
- `REMOTE_REPOSITORY_BSUIT`: Repository for Bilibili suit images
- `REMOTE_REPOSITORY_X`: Repository for artist database
- `HF_TOKEN_X`: Alternative HuggingFace token for artist database

## Main Commands

### CLI Entry Point
```bash
# Process newest N posts from Skeb.jp (default: 200)
python -m test.prepare newest -n 200

# Repack unarchived zips into larger packs (max 5.5GB)
python -m test.prepare pack

# Push artist database to HuggingFace
python -m test.prepare artists
```

### Direct Script Execution
```bash
# Crawl Bilibili suit images
python test/prepare/bsuit/crawler.py

# Crawl Bilibili act images
python test/prepare/bact/crawler.py
```

### Testing
```bash
# Run all unit tests
make test

# Run tests for specific directory
RANGE_DIR=client make unittest

# Run tests with coverage
make unittest COV_TYPES="xml term-missing"
```

## Architecture

### Core Workflow (`test/prepare/`)

**1. Listing & URL Extraction** (`listing.py`, `url.py`)
- `list_newest_posts()`: Fetches recent Skeb.jp posts via SkebClient
- `get_urls_from_post()`: Extracts URLs from post body/source_body
- `extract_urls()`: Regex-based URL extraction from text

**2. URL Processing** (`process.py`)
- `try_process_url()`: Main entry point for processing URLs
- Detects URL type (Google Drive, Imgur, Dropbox) using `KNOWN_SITES` list
- Downloads content to temporary directory
- Creates zip archive with sanitized filenames (prefix + underscored name)
- Uploads to HuggingFace dataset under `unarchived/` directory
- Tracks processed resource IDs in `archived.json` to avoid duplicates

**3. Download Handlers**
- `google.py`: Google Drive files/folders using `gdown` library with rate limiting
- `imgur.py`: Imgur albums via API (extracts client_id from main.js)
- `dropbox.py`: Dropbox shared links (converts `dl=0` to `dl=1`)

**4. Repacking** (`repack.py`)
- `repack_all()`: Consolidates small zips from `unarchived/` into larger packs
- Max pack size: 5.5GB (configurable)
- Moves files to `packs/` directory
- Updates `index.json` with pack metadata
- Generates README.md with download links table
- Deletes source files from `unarchived/` after successful pack creation

**5. Batch Processing** (`lololo.py`)
- `batch_process_newest()`: Iterates through newest posts with rate limiting (4s default)
- Error handling with 10s sleep on crashes
- Calls `try_process_url()` for each extracted URL

**6. Specialized Crawlers**
- `bsuit/crawler.py`: Bilibili mall suit background images
  - Uses Bilibili API with SPI authentication
  - Crawls space_bg items with portrait images
  - Tracks processed suit_ids to avoid duplicates
- `bact/crawler.py`: Similar structure for Bilibili act images
- `artists_idx.py`: Danbooru artist database scraper
  - Fetches all artists from Danbooru API (1000/page)
  - Cross-references with tag post counts from SQLite database
  - Creates SQLite database with artists and aliases tables
  - Uploads to HuggingFace dataset

### Supporting Infrastructure

**SkebClient** (`pyskeb/client/client.py`)
- Minimal Skeb.jp API wrapper
- Handles 429 rate limiting via `request_key` cookie extraction
- Methods: `iter_art_pages()`, `get_post()`, `iter_user_pages()`, `iter_work_pages()`

**Base Configuration** (`test/prepare/base.py`)
- HuggingFace client/filesystem initialization
- Repository creation with LFS configuration
- `number_to_tag()`: Converts numbers to size range tags (n<1K, 1K<n<10K, etc.)
- `make_index_file()`: Creates tar index with hashes

**Session Management** (`pyskeb/utils/session.py`)
- `get_requests_session()`: Retry logic, timeouts, random UA rotation
- `srequest()`: Request wrapper with exponential backoff

## Data Flow

1. **Scrape**: `newest` command → Skeb.jp posts → extract URLs
2. **Download**: URLs → detect type → download to temp dir → zip
3. **Upload**: Zip → HuggingFace `unarchived/` directory
4. **Repack**: `pack` command → consolidate zips → `packs/` directory
5. **Track**: Update `archived.json` and `index.json` metadata

## Key Design Patterns

- **Resource ID tracking**: Each URL generates unique resource_id (e.g., `googledrive_{file_id}`, `imgur_{album_id}`)
- **Idempotency**: Checks `archived.json` and `unarchived/` before processing
- **Filename sanitization**: Replaces non-alphanumeric chars with underscores
- **Rate limiting**: Wait times between requests to avoid bans
- **Error resilience**: Try-except blocks with logging, continues on failures
- **Atomic commits**: HuggingFace operations use commit operations for consistency

## GitHub Actions Workflows

- `newest.yml`: Scheduled crawling of newest Skeb posts
- `repack.yml`: Periodic repacking of unarchived files
- `artists.yml`: Artist database updates
- `n_bsuit.yml`, `n_bsuit_v.yml`: Bilibili suit crawling
- `n_bact.yml`, `n_bact_v.yml`: Bilibili act crawling

## Dependencies

**Core:**
- `hbutils`: System utilities, URL parsing
- `requests`: HTTP client
- `huggingface_hub`: Dataset uploads
- `gdown`: Google Drive downloads
- `pyquery`: HTML parsing for Imgur
- `pandas`: Data manipulation for metadata

**Testing:**
- `pytest` with plugins: cov, mock, xdist, rerunfailures, timeout, benchmark
