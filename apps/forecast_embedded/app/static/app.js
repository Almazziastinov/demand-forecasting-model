document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.querySelector("[data-loading-overlay]");
  const overlayTitle = overlay?.querySelector("[data-loading-title]");
  const overlayMessage = overlay?.querySelector("[data-loading-message]");
  let downloadHideTimer = null;

  const showLoading = (title = "Загрузка данных", message = "Подождите, действие выполняется.") => {
    if (!overlay) {
      return;
    }
    if (downloadHideTimer) {
      window.clearTimeout(downloadHideTimer);
      downloadHideTimer = null;
    }
    if (overlayTitle) {
      overlayTitle.textContent = title;
    }
    if (overlayMessage) {
      overlayMessage.textContent = message;
    }
    overlay.hidden = false;
    document.body.classList.add("is-loading");
  };

  const hideLoading = () => {
    if (!overlay) {
      return;
    }
    overlay.hidden = true;
    document.body.classList.remove("is-loading");
  };

  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      showLoading();
    });
    form.querySelectorAll("input, select").forEach((control) => {
      control.addEventListener("change", () => {
        showLoading();
      });
    });
  });

  document.querySelectorAll("a[href]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.defaultPrevented || event.button !== 0 || link.target === "_blank") {
        return;
      }
      const href = link.getAttribute("href") || "";
      if (!href || href.startsWith("#") || href.startsWith("javascript:")) {
        return;
      }

      if (href.includes("baking-plan.xlsx")) {
        showLoading("Формируем план выпекания", "Excel-файл готовится и скоро начнёт скачиваться.");
        downloadHideTimer = window.setTimeout(hideLoading, 10000);
        return;
      }

      showLoading();
    });
  });

  window.addEventListener("pageshow", hideLoading);
  window.addEventListener("focus", () => {
    if (downloadHideTimer) {
      window.setTimeout(hideLoading, 600);
    }
  });

  document.querySelectorAll("[data-bakery-search]").forEach((input) => {
    const sidebar = input.closest(".sidebar");
    const links = Array.from(sidebar?.querySelectorAll(".bakery-link") || []);
    const clearButton = sidebar?.querySelector("[data-bakery-search-clear]");
    const emptyNote = sidebar?.querySelector("[data-bakery-search-empty]");
    const meta = sidebar?.querySelector("[data-bakery-search-meta]");
    const total = links.length;

    const normalize = (value) =>
      value
        .toLocaleLowerCase("ru-RU")
        .replaceAll("ё", "е")
        .replace(/\s+/g, " ")
        .trim();

    const updateMeta = (visibleCount, query) => {
      if (!meta) {
        return;
      }
      meta.textContent = query ? `${visibleCount} из ${total}` : `${total} пекарен`;
    };

    const applyFilter = () => {
      const query = normalize(input.value);
      let visibleCount = 0;

      links.forEach((link) => {
        const text = normalize(link.textContent);
        const isVisible = query.length === 0 || text.includes(query);
        link.classList.toggle("is-search-hidden", !isVisible);
        if (isVisible) {
          visibleCount += 1;
        }
      });

      if (emptyNote) {
        emptyNote.hidden = visibleCount > 0 || query.length === 0;
      }
      if (clearButton) {
        clearButton.hidden = query.length === 0;
      }
      updateMeta(visibleCount, query);
    };

    input.addEventListener("input", applyFilter);
    clearButton?.addEventListener("click", () => {
      input.value = "";
      input.focus();
      applyFilter();
    });
    applyFilter();
  });

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const formatQty = (value) => Math.round(Number(value || 0)).toLocaleString("ru-RU");
  const formatDelta = (value) => {
    const rounded = Math.round(Number(value || 0));
    return `${rounded > 0 ? "+" : ""}${rounded.toLocaleString("ru-RU")}`;
  };
  const formatPct = (value) => `${Math.round(Number(value || 0) * 100)}%`;

  document.querySelectorAll("[data-hour-discrepancy-root]").forEach((root) => {
    const panel = root.parentElement?.querySelector("[data-hour-discrepancy-panel]");
    const bakeryId = root.dataset.bakeryId;
    const date = root.dataset.date;
    const runId = root.dataset.runId;

    if (!panel || !bakeryId || !date) {
      return;
    }

    const renderLoading = (hour) => {
      panel.hidden = false;
      panel.innerHTML = `
        <div class="hour-discrepancy-head">
          <strong>${String(hour).padStart(2, "0")}:00</strong>
          <span>Загружаем SKU-вклад...</span>
        </div>
      `;
    };

    const renderError = (hour) => {
      panel.hidden = false;
      panel.innerHTML = `
        <div class="hour-discrepancy-head">
          <strong>${String(hour).padStart(2, "0")}:00</strong>
          <span>Не удалось загрузить детализацию.</span>
        </div>
      `;
    };

    const renderRows = (payload) => {
      const hour = String(payload.hour).padStart(2, "0");
      const delta = Number(payload.delta || 0);
      const directionText =
        delta > 0 ? "прогноз выше факта" : delta < 0 ? "прогноз ниже факта" : "расхождение около нуля";
      const items = Array.isArray(payload.items) ? payload.items : [];
      const rowsHtml = items
        .map(
          (item) => `
            <div class="hour-discrepancy-row">
              <span>
                <strong>${escapeHtml(item.product_name || "SKU без названия")}</strong>
                <small>${escapeHtml(item.category_name || "Без группы")}</small>
              </span>
              <span>${formatQty(item.actual_qty)}</span>
              <span>${formatQty(item.forecast_qty)}</span>
              <span class="delta">${formatDelta(item.delta)}</span>
              <span>${formatPct(item.contribution_pct)}</span>
            </div>
          `,
        )
        .join("");

      panel.hidden = false;
      panel.innerHTML = `
        <div class="hour-discrepancy-head">
          <strong>${hour}:00 — ${directionText} на ${formatDelta(delta)} шт.</strong>
          <span>SKU, которые сильнее всего объясняют этот час</span>
        </div>
        ${
          items.length
            ? `<div class="hour-discrepancy-table">
                <div class="hour-discrepancy-row header">
                  <span>SKU</span>
                  <span>Факт</span>
                  <span>Прогноз</span>
                  <span>Разница</span>
                  <span>Вклад</span>
                </div>
                ${rowsHtml}
              </div>`
            : `<div class="empty-note">Нет SKU с заметным вкладом в выбранном направлении.</div>`
        }
      `;
    };

    root.querySelectorAll(".hour-card").forEach((button) => {
      button.addEventListener("click", async () => {
        const hour = button.dataset.hour;
        if (!hour) {
          return;
        }

        root.querySelectorAll(".hour-card").forEach((item) => item.classList.remove("is-selected"));
        button.classList.add("is-selected");
        renderLoading(hour);

        const params = new URLSearchParams({ date, hour, limit: "10" });
        if (runId) {
          params.set("run_id", runId);
        }

        try {
          const response = await fetch(`/api/v1/bakeries/${bakeryId}/hour-discrepancy?${params.toString()}`, {
            headers: { Accept: "application/json" },
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          renderRows(await response.json());
        } catch (error) {
          renderError(hour);
        }
      });
    });
  });
});
