import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WebSocketClient, type WebSocketLike } from './websocket';

class FakeSocket implements WebSocketLike {
  public readyState: number = WebSocket.CONNECTING;

  public onopen: ((event: Event) => void) | null = null;

  public onmessage: ((event: MessageEvent<string>) => void) | null = null;

  public onclose: ((event: CloseEvent) => void) | null = null;

  public onerror: ((event: Event) => void) | null = null;

  public readonly sentMessages: string[] = [];

  public constructor(public readonly url: string) {}

  public send(data: string): void {
    this.sentMessages.push(data);
  }

  public close(): void {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close'));
  }

  public open(): void {
    this.readyState = WebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  public receive(data: unknown): void {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }));
  }
}

describe('WebSocketClient', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('connects with API key query auth and dispatches topic messages', () => {
    const sockets: FakeSocket[] = [];
    const handler = vi.fn();
    const client = new WebSocketClient({
      url: 'ws://api.test/ws',
      apiKey: 'local-key',
      clientId: 'client-1',
      socketFactory: (url) => {
        const socket = new FakeSocket(url);
        sockets.push(socket);
        return socket;
      },
      logger: { debug: vi.fn(), error: vi.fn(), warn: vi.fn() },
    });

    const unsubscribe = client.subscribe('training:model-1', handler);
    expect(sockets[0]?.url).toBe('ws://api.test/ws?api_key=local-key&client_id=client-1');

    sockets[0]?.open();
    expect(sockets[0]?.sentMessages).toContain(
      JSON.stringify({ type: 'subscribe', topic: 'training:model-1' }),
    );

    sockets[0]?.receive({ topic: 'training:model-1', data: { progress_percent: 24 } });
    expect(handler).toHaveBeenCalledWith({ progress_percent: 24 });

    unsubscribe();
    expect(sockets[0]?.sentMessages).toContain(
      JSON.stringify({ type: 'unsubscribe', topic: 'training:model-1' }),
    );

    client.disconnect();
  });

  it('reconnects and resubscribes after an unexpected close', () => {
    const sockets: FakeSocket[] = [];
    const client = new WebSocketClient({
      url: 'ws://api.test/ws',
      maxReconnectAttempts: 2,
      reconnectBaseDelayMs: 10,
      socketFactory: (url) => {
        const socket = new FakeSocket(url);
        sockets.push(socket);
        return socket;
      },
      logger: { debug: vi.fn(), error: vi.fn(), warn: vi.fn() },
    });

    client.subscribe('backtest:job-1', vi.fn());
    sockets[0]?.open();
    sockets[0]?.close();

    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(10);
    expect(sockets).toHaveLength(2);

    sockets[1]?.open();
    expect(sockets[1]?.sentMessages).toContain(
      JSON.stringify({ type: 'subscribe', topic: 'backtest:job-1' }),
    );

    client.disconnect();
  });
});