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
});
