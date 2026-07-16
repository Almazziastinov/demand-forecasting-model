import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath =
  process.argv[2] ??
  "C:/Users/dns/Documents/Codex/2026-06-30/new-chat/work/reviews_sentiment/input.xlsx";
const outputPath =
  process.argv[3] ??
  "C:/Users/dns/Documents/Codex/2026-06-30/new-chat/outputs/Отзывы покупателей нормализованные категории.xlsx";

const REVIEW_SHEETS = new Set(["Отзывы 2025", "2024-25 Жалобы", "2022 Полож отзывы"]);
const CATEGORY_HEADERS = new Set(["Категория", "Тема сообщения"]);
const NORMALIZED_HEADER = "нормализованная категория";

function normalizeCategory(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "";

  const key = raw.toLowerCase().replaceAll("ё", "е").replace(/\s+/g, " ");

  const dictionary = new Map([
    ["благодарность", "Благодарность"],
    ["благодарсность", "Благодарность"],
    ["сервис", "Сервис"],
    ["качество продукции", "Качество продукции"],
    ["качество", "Качество продукции"],
    ["качество продукта", "Качество продукции"],
    ["качество сырья", "Качество продукции"],
    ["качкство", "Качество продукции"],
    ["состав", "Качество продукции"],
    ["вкус", "Качество продукции"],
    ["упаковка", "Качество продукции"],
    ["хот доги", "Качество продукции"],
    ["кбжу", "Качество продукции"],
    ["чистота", "Чистота"],
    ["просрок", "Просрочка"],
    ["просрочка", "Просрочка"],
    ["звонок на горячую линию", "Звонок на горячую линию"],
    ["отзыв бывшего сотрудника", "Отзыв сотрудника"],
    ["отзыв сотрудника", "Отзыв сотрудника"],
    ["жалоба персонала", "Жалоба персонала"],
    ["помещение", "Помещение"],
    ["ассортимент", "Ассортимент"],
  ]);

  return dictionary.get(key) ?? raw;
}

function countBy(values) {
  const counts = new Map();
  for (const value of values) {
    if (!value) continue;
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ru"));
}

function getHeaderValues(sheet) {
  const used = sheet.getUsedRange(true);
  return sheet.getRangeByIndexes(0, 0, 1, used.columnCount).values[0];
}

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summaryRows = [["Лист", "Исходная категория", "Нормализованная категория", "Количество"]];

for (const sheet of workbook.worksheets.items) {
  if (!REVIEW_SHEETS.has(sheet.name)) continue;

  const used = sheet.getUsedRange(true);
  const rowCount = used.rowCount;
  const headers = getHeaderValues(sheet);
  const categoryCol = headers.findIndex((header) => CATEGORY_HEADERS.has(String(header ?? "").trim()));
  if (categoryCol < 0) continue;

  const existingNormalizedCol = headers.findIndex(
    (header) => String(header ?? "").trim().toLowerCase() === NORMALIZED_HEADER,
  );
  const outputCol = existingNormalizedCol >= 0 ? existingNormalizedCol : headers.length;

  const categoryValues = sheet.getRangeByIndexes(1, categoryCol, rowCount - 1, 1).values;
  const normalizedValues = categoryValues.map(([value]) => [normalizeCategory(value)]);

  const headerCell = sheet.getCell(0, outputCol);
  headerCell.values = [[NORMALIZED_HEADER]];
  headerCell.format = {
    fill: "#D9EAD3",
    font: { bold: true, color: "#1F2937" },
    alignment: { horizontal: "center", vertical: "center" },
  };

  const outputRange = sheet.getRangeByIndexes(1, outputCol, normalizedValues.length, 1);
  outputRange.values = normalizedValues;
  outputRange.format = { alignment: { horizontal: "center", vertical: "center" } };
  sheet.getRangeByIndexes(0, outputCol, normalizedValues.length + 1, 1).format.columnWidthPx = 190;
  sheet.freezePanes.freezeRows(1);

  const pairCounts = new Map();
  for (let i = 0; i < categoryValues.length; i += 1) {
    const source = String(categoryValues[i][0] ?? "").trim();
    const normalized = normalizedValues[i][0];
    if (!source && !normalized) continue;
    const key = `${source}\u0000${normalized}`;
    pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
  }
  for (const [key, count] of [...pairCounts.entries()].sort((a, b) => b[1] - a[1])) {
    const [source, normalized] = key.split("\u0000");
    summaryRows.push([sheet.name, source, normalized, count]);
  }
}

const summarySheet = workbook.worksheets.add("Нормализация категорий");
summarySheet.getRangeByIndexes(0, 0, summaryRows.length, 4).values = summaryRows;
summarySheet.getRange("A1:D1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  alignment: { horizontal: "center", vertical: "center" },
};
summarySheet.getRangeByIndexes(0, 0, summaryRows.length, 4).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2F3",
};
summarySheet.getRange("A:A").format.columnWidthPx = 170;
summarySheet.getRange("B:B").format.columnWidthPx = 210;
summarySheet.getRange("C:C").format.columnWidthPx = 230;
summarySheet.getRange("D:D").format.columnWidthPx = 100;
summarySheet.freezePanes.freezeRows(1);

const normalizedOnly = summaryRows.slice(1).map((row) => row[2]);
console.log(JSON.stringify(Object.fromEntries(countBy(normalizedOnly)), null, 2));

const outputDir = path.dirname(outputPath);
await fs.mkdir(outputDir, { recursive: true });

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 2000,
});
console.log(errorScan.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved: ${outputPath}`);
