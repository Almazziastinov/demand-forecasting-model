import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = process.argv[2] ?? "C:/Users/dns/Downloads/Отзывы покупателей.xlsx";
const resultsPath =
  process.argv[3] ??
  "C:/Users/dns/Desktop/Projects/demand-forecasting-model/outputs/buyer_reviews_sentiment_results.json";
const outputPath =
  process.argv[4] ??
  "C:/Users/dns/Documents/Codex/2026-06-30/new-chat/outputs/Отзывы покупателей с тональностью.xlsx";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const results = JSON.parse(await fs.readFile(resultsPath, "utf8"));

const bySheet = new Map();
for (const result of results) {
  if (!bySheet.has(result.sheet)) {
    bySheet.set(result.sheet, []);
  }
  bySheet.get(result.sheet).push(result);
}

function getHeaderValues(sheet) {
  const used = sheet.getUsedRange(true);
  const colCount = used.columnCount;
  return sheet.getRangeByIndexes(0, 0, 1, colCount).values[0];
}

for (const [sheetName, sheetResults] of bySheet.entries()) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const headers = getHeaderValues(sheet);
  const existingIndex = headers.findIndex(
    (value) => String(value ?? "").trim().toLowerCase() === "тональность",
  );
  const targetColZeroBased =
    existingIndex >= 0 ? existingIndex : headers.findIndex((value) => value === null || value === "");
  const col = targetColZeroBased >= 0 ? targetColZeroBased : headers.length;

  const headerCell = sheet.getCell(0, col);
  headerCell.values = [["тональность"]];
  headerCell.format = {
    fill: "#D9EAD3",
    font: { bold: true, color: "#1F2937" },
    alignment: { horizontal: "center", vertical: "center" },
  };

  const maxRow = Math.max(...sheetResults.map((result) => result.row));
  const values = Array.from({ length: maxRow - 1 }, () => [""]);
  for (const result of sheetResults) {
    values[result.row - 2] = [result.sentiment];
  }

  const targetRange = sheet.getRangeByIndexes(1, col, values.length, 1);
  targetRange.values = values;
  targetRange.format = {
    alignment: { horizontal: "center", vertical: "center" },
  };

  sheet.getRangeByIndexes(0, col, values.length + 1, 1).format.columnWidthPx = 120;
  sheet.freezePanes.freezeRows(1);
}

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
