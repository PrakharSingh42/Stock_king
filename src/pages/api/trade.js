import connectToDatabase from "@/libs/mongodb";
import Trade from "@/models/trade";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method Not Allowed" });
  }

  try {
    const { ticker, quantity, action, price } = req.body;

    if (!ticker || !quantity || !action || !price) {
      return res.status(400).json({ message: "Missing required fields" });
    }
    if (!["buy", "sell"].includes(action)) {
      return res.status(400).json({ message: "Invalid action type" });
    }

    await connectToDatabase();

    const trade = new Trade({
      ticker,
      quantity,
      action,
      price,
    });

    await trade.save();

    return res.status(200).json({ message: "Trade saved", trade });
  } catch (error) {
    console.error("Trade Error:", error);
    return res.status(500).json({ message: "Internal Server Error" });
  }
}
