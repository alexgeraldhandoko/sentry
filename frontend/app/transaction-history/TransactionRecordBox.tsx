import { Cormorant_Garamond } from "next/font/google";

const garamond = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

type TransactionRecordBoxProps = {
  classification: string;
  confidence: number;
  trueLabel: string;
  timestamp: number;
};

export default function TransactionRecordBox({
  classification,
  confidence,
  trueLabel,
  timestamp,
}: TransactionRecordBoxProps) {
  const isFraudulent = classification.toLowerCase().includes("fraud");

  const formattedTimestamp = new Date(timestamp * 1000).toLocaleString();

  return (
    <article className="border border-[#c8b18a] bg-[#fffaf2] p-5 transition hover:bg-[#fff7ec]">
      <div className="flex flex-col gap-3 border-b border-[#d8c6aa] pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[#8a6b3d]">
            Transaction Record
          </p>

          <h3
            className={`${garamond.className} mt-2 text-2xl font-semibold ${
              isFraudulent ? "text-[#6b1f2a]" : "text-[#3f4f2e]"
            }`}
          >
            {classification}
          </h3>
        </div>

        <div className="border border-[#d8c6aa] bg-[#fffdf8] px-3 py-2 text-right">
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#8a6b3d]">
            Confidence
          </p>

          <p
            className={`${garamond.className} mt-1 text-xl font-semibold text-[#2b211b]`}
          >
            {Number(confidence).toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="border border-[#d8c6aa] bg-[#fffdf8] p-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#8a6b3d]">
            True Label
          </p>

          <p className="mt-1 text-sm font-medium text-[#3a2d24]">
            {trueLabel}
          </p>
        </div>

        <div className="border border-[#d8c6aa] bg-[#fffdf8] p-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#8a6b3d]">
            Timestamp
          </p>

          <p className="mt-1 text-sm font-medium text-[#3a2d24]">
            {formattedTimestamp}
          </p>
        </div>
      </div>
    </article>
  );
}