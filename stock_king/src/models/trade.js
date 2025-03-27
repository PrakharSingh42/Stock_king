import mongoose from "mongoose";

const TradeSchema = new mongoose.Schema({
  ticker: { type: String, required: true },
  quantity: { type: Number, required: true },
  action: { type: String, enum: ["buy", "sell"], required: true },
  price: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now },
});

export default mongoose.models.Trade || mongoose.model("Trade", TradeSchema);
