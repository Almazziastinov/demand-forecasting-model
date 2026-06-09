document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-bakery-search]").forEach((input) => {
    const sidebar = input.closest(".sidebar");
    const links = Array.from(sidebar?.querySelectorAll(".bakery-link") || []);

    input.addEventListener("input", () => {
      const query = input.value.trim().toLocaleLowerCase("ru-RU");
      links.forEach((link) => {
        const text = link.textContent.toLocaleLowerCase("ru-RU");
        link.hidden = query.length > 0 && !text.includes(query);
      });
    });
  });
});
