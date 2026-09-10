.DEFAULT_GOAL := help

# Colors
G := \033[0;32m
Y := \033[1;33m
R := \033[0;31m
C := \033[0;36m
B := \033[1m
X := \033[0m

# Auto-detect local IP (en0 = WiFi on Mac)
LOCAL_IP := $(shell ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
CERT_DIR  := .certs

.PHONY: up down restart build logs open shell status mobile-setup mobile train clean nuke help

##@ Start / Stop

up: ## Build image if needed, then start the web UI
	@echo "$(B)$(G)→ Wake Word Trainer$(X)"
	@docker compose up --build -d web
	@echo ""
	@echo "  $(G)✓ Ready$(X)  $(C)http://localhost:3000$(X)"
	@echo "  Logs → $(B)make logs$(X)    Stop → $(B)make down$(X)"
	@echo ""

down: ## Stop all services
	@echo "$(Y)→ Stopping...$(X)"
	@docker compose down
	@echo "$(G)✓ Stopped$(X)"

restart: down up ## Stop, then start again

##@ Development

build: ## Force full rebuild without cache
	@echo "$(Y)→ Rebuilding (no cache)...$(X)"
	@docker compose build --no-cache web
	@echo "$(G)✓ Done — run$(X) $(B)make up$(X)"

logs: ## Follow live logs  (Ctrl+C to exit)
	@docker compose logs -f --tail=100 web

open: ## Open http://localhost:3000 in the browser
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000

shell: ## Open a bash shell inside the running container
	@docker compose exec web bash

status: ## Show container status
	@docker compose ps

##@ iPhone / Mobile (lokales HTTPS, kein Internet nötig)

mobile-setup: ## Einmalig: lokales HTTPS-Zertifikat erstellen + iPhone-Anleitung
	@command -v mkcert >/dev/null 2>&1 || { \
		echo "$(Y)→ mkcert installieren...$(X)"; \
		brew install mkcert; }
	@mkcert -install
	@mkdir -p $(CERT_DIR)
	@cd $(CERT_DIR) && mkcert $(LOCAL_IP) localhost 127.0.0.1
	@echo ""
	@echo "$(G)✓ Zertifikat erstellt für $(B)$(LOCAL_IP)$(X)"
	@echo ""
	@echo "$(B)iPhone einrichten (einmalig):$(X)"
	@echo ""
	@echo "  $(Y)Schritt 1:$(X) Root-Zertifikat per AirDrop an dein iPhone senden:"
	@echo "  $(B)open $$(mkcert -CAROOT)$(X)"
	@echo "  → Datei $(B)rootCA.pem$(X) per AirDrop ans iPhone schicken"
	@echo ""
	@echo "  $(Y)Schritt 2:$(X) Auf dem iPhone:"
	@echo "  Einstellungen → Allgemein → VPN & Geräteverwaltung → Zertifikat installieren"
	@echo ""
	@echo "  $(Y)Schritt 3:$(X) Auf dem iPhone:"
	@echo "  Einstellungen → Allgemein → Info → Zertifikatsvertrauenseinstellungen"
	@echo "  → $(B)mkcert$(X) aktivieren"
	@echo ""
	@echo "Danach: $(B)make mobile$(X)"

mobile: ## Lokalen HTTPS-Proxy starten — iPhone öffnet https://$(LOCAL_IP):3443
	@[ -f "$(CERT_DIR)/$(LOCAL_IP)+2.pem" ] || \
	 [ -f "$(CERT_DIR)/$(LOCAL_IP)+2-key.pem" ] || { \
		echo "$(R)Kein Zertifikat gefunden.$(X) Bitte zuerst: $(B)make mobile-setup$(X)"; \
		exit 1; }
	@echo ""
	@echo "$(B)$(G)→ HTTPS-Proxy läuft$(X)"
	@echo ""
	@echo "  Auf dem iPhone öffnen: $(B)$(C)https://$(LOCAL_IP):3443$(X)"
	@echo "  (iPhone und Mac müssen im selben WLAN sein)"
	@echo "  $(Y)Strg+C$(X) zum Beenden"
	@echo ""
	@cd $(CERT_DIR) && npx --yes local-ssl-proxy \
		--source 3443 --target 3000 \
		--cert "$(LOCAL_IP)+2.pem" \
		--key  "$(LOCAL_IP)+2-key.pem" \
		--hostname 0.0.0.0

##@ CLI Training  (no web UI required)

prefetch: ## Pre-download all training data (~13 GB) into ./data — run once before --full
	@echo "$(Y)→ Pre-fetching all training data into ./data/ ...$(X)"
	@echo "$(Y)  MIT RIRs · AudioSet · MUSAN · validation features · ACAV100M (~11 GB)$(X)"
	@echo ""
	@docker compose run --rm -e PYTHONUNBUFFERED=1 trainer --prefetch
	@echo ""
	@echo "$(G)✓ All data cached. Run$(X) $(B)make train WORD=\"Hey Dobbi\" FULL=1$(X) $(G)to start immediately.$(X)"

train: ## Run training — make train WORD="Hey Dobbi" [SAMPLES=2000] [STEPS=25000] [FULL=1] [LANG=de]
	@[ -n "$(WORD)" ] || { \
		echo "$(R)Error: WORD is required$(X)"; \
		echo "Usage: $(B)make train WORD=\"Hey Dobbi\"$(X)"; \
		echo "       $(B)make train WORD=\"Hey Dobbi\" SAMPLES=1000 STEPS=10000 LANG=de$(X)"; \
		exit 1; }
	@docker compose run --rm \
		-e PYTHONUNBUFFERED=1 \
		-e SPEAKY_LANG=$(or $(LANG),de) \
		trainer "$(WORD)" \
		--samples $(or $(SAMPLES),2000) \
		--steps   $(or $(STEPS),25000) \
		--lang    $(or $(LANG),de) \
		$(if $(FULL),--full,)

##@ Cleanup

clean: ## Remove containers, image and DB — keeps ./data and ./output
	@echo "$(Y)→ Removing containers, image, DB volume...$(X)"
	@docker compose down --rmi local --volumes --remove-orphans
	@echo "$(G)✓ Clean. Run$(X) $(B)make$(X) $(G)to rebuild.$(X)"

nuke: ## ⚠ FULL RESET — removes everything including all downloaded data
	@echo ""
	@echo "$(R)$(B)  ⚠  FULL RESET$(X)"
	@echo "$(R)  Removes: containers · images · volumes$(X)"
	@echo "$(R)           ./data  ./output  ./piper-voices$(X)"
	@echo ""
	@printf "$(Y)  Type YES to confirm: $(X)" && read c && [ "$$c" = "YES" ] || { echo "Aborted."; exit 1; }
	@docker compose down --rmi all --volumes --remove-orphans
	@rm -rf data output piper-voices
	@echo "$(G)✓ Full reset done. Run$(X) $(B)make$(X) $(G)to start from scratch.$(X)"

##@ Help

help: ## Show this help (default)
	@echo ""
	@echo "$(B)  Wake Word Trainer$(X)"
	@echo ""
	@awk ' \
		/^##@/ { printf "\n$(B)  %s$(X)\n", substr($$0, 5) } \
		/^[a-z][a-zA-Z_-]+:.*## / { \
			split($$0, a, ":.*## "); \
			printf "  $(G)make %-14s$(X) %s\n", a[1], a[2] \
		}' $(MAKEFILE_LIST)
	@echo ""
