import { useEffect, useState } from "react";

const FINNHUB_API_KEY = "cvcq8ipr01qodeuati6gcvcq8ipr01qodeuati70"; // Replace with your API key

const useStockWebSocket = (ticker: string) => {
  const [price, setPrice] = useState<number | null>(null);

  useEffect(() => {
    const socket = new WebSocket(`wss://ws.finnhub.io?token=${FINNHUB_API_KEY}`);

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "subscribe", symbol: ticker }));
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "trade" && data.data.length > 0) {
        setPrice(data.data[0].p); // Extract the price from the WebSocket response
      }
    };

    return () => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "unsubscribe", symbol: ticker }));
      }
      socket?.close();
    };
    
  }, [ticker]);

  return price;
};

export default useStockWebSocket;
