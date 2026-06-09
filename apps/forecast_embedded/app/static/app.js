document.addEventListener("DOMContentLoaded", () => {
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
