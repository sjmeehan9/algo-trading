const DEFAULT_WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';

export type MessageHandler<T = unknown> = (data: T) => void;

export interface WebSocketLike {
  readonly readyState: number;
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent<string>) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  send(data: string): void;
  close(): void;
}

export type WebSocketFactory = (url: string) => WebSocketLike;

export interface WebSocketClientOptions {
  url?: string;
  apiKey?: string;
  clientId?: string;
  maxReconnectAttempts?: number;
  reconnectBaseDelayMs?: number;
  reconnectMaxDelayMs?: number;
  socketFactory?: WebSocketFactory;
  logger?: Pick<Console, 'debug' | 'error' | 'warn'>;
}

interface TopicMessage<T = unknown> {
  topic?: string;
  data?: T;
  type?: string;
}

const browserSocketFactory: WebSocketFactory = (url: string) => new WebSocket(url);

const appendAuthParams = (url: string, apiKey: string, clientId?: string): string => {
  const parsedUrl = new URL(url);

  if (apiKey) {
    parsedUrl.searchParams.set('api_key', apiKey);
  }

  if (clientId) {
    parsedUrl.searchParams.set('client_id', clientId);
  }

  return parsedUrl.toString();
};

/** Topic-aware WebSocket client with automatic reconnect and resubscription. */
export class WebSocketClient {
  private socket: WebSocketLike | null = null;

  private readonly url: string;

  private readonly apiKey: string;

  private readonly clientId?: string;

  private readonly maxReconnectAttempts: number;

  private readonly reconnectBaseDelayMs: number;

  private readonly reconnectMaxDelayMs: number;

  private readonly socketFactory: WebSocketFactory;

  private readonly logger: Pick<Console, 'debug' | 'error' | 'warn'>;

  private readonly handlers = new Map<string, Set<MessageHandler>>();

  private readonly subscriptions = new Set<string>();

  private reconnectAttempts = 0;

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private shouldReconnect = true;

  public constructor(options: WebSocketClientOptions = {}) {
    this.url = options.url || DEFAULT_WS_URL;
    this.apiKey = options.apiKey ?? DEFAULT_API_KEY;
    this.clientId = options.clientId;
    this.maxReconnectAttempts = options.maxReconnectAttempts ?? 5;
    this.reconnectBaseDelayMs = options.reconnectBaseDelayMs ?? 1_000;
    this.reconnectMaxDelayMs = options.reconnectMaxDelayMs ?? 30_000;
    this.socketFactory = options.socketFactory || browserSocketFactory;
    this.logger = options.logger || console;
  }

  /** Establish a WebSocket connection if one is not already open or opening. */
  public connect(): void {
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    this.shouldReconnect = true;
    this.clearReconnectTimer();

    const socketUrl = appendAuthParams(this.url, this.apiKey, this.clientId);
    const socket = this.socketFactory(socketUrl);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.logger.debug('WebSocket connected.');
      this.subscriptions.forEach((topic) => this.sendControlMessage('subscribe', topic));
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      this.handleMessage(event.data);
    };

    socket.onerror = (event: Event) => {
      this.logger.error('WebSocket error.', event);
    };

    socket.onclose = () => {
      this.socket = null;
      this.logger.debug('WebSocket disconnected.');
      this.scheduleReconnect();
    };
  }

  /** Subscribe a handler to a backend topic and return an unsubscribe function. */
  public subscribe<T>(topic: string, handler: MessageHandler<T>): () => void {
    const typedHandler = handler as MessageHandler;

    if (!this.handlers.has(topic)) {
      this.handlers.set(topic, new Set());
    }

    this.handlers.get(topic)?.add(typedHandler);
    this.subscriptions.add(topic);
    this.connect();

    if (this.socket?.readyState === WebSocket.OPEN) {
      this.sendControlMessage('subscribe', topic);
    }

    return () => {
      const topicHandlers = this.handlers.get(topic);
      topicHandlers?.delete(typedHandler);

      if (!topicHandlers || topicHandlers.size === 0) {
        this.handlers.delete(topic);
        this.subscriptions.delete(topic);

        if (this.socket?.readyState === WebSocket.OPEN) {
          this.sendControlMessage('unsubscribe', topic);
        }
      }
    };
  }

  /** Close the current WebSocket connection and prevent automatic reconnects. */
  public disconnect(): void {
    this.shouldReconnect = false;
    this.clearReconnectTimer();

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  private handleMessage(rawData: string): void {
    let message: TopicMessage;

    try {
      message = JSON.parse(rawData) as TopicMessage;
    } catch (error) {
      this.logger.warn('Received malformed WebSocket message.', error);
      return;
    }

    if (!message.topic) {
      return;
    }

    const handlers = this.handlers.get(message.topic);
    handlers?.forEach((handler) => handler(message.data));
  }

  private sendControlMessage(type: 'subscribe' | 'unsubscribe', topic: string): void {
    this.socket?.send(JSON.stringify({ type, topic }));
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.reconnectAttempts += 1;
    const delayMs = Math.min(
      this.reconnectBaseDelayMs * 2 ** (this.reconnectAttempts - 1),
      this.reconnectMaxDelayMs,
    );

    this.reconnectTimer = setTimeout(() => this.connect(), delayMs);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

export const wsClient = new WebSocketClient();