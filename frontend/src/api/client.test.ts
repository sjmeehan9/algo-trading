import { describe, expect, it } from 'vitest';

import { ApiClient, type PaginatedResponse } from './client';

import type { AxiosAdapter } from 'axios';

const createAdapter = (data: unknown, status = 200): AxiosAdapter =>
  async (config) => ({
    config,
    data,
    headers: {},
    request: {},
    status,
    statusText: status >= 400 ? 'Error' : 'OK',
  });

describe('ApiClient', () => {
  it('unwraps standardized API responses', async () => {
    const client = new ApiClient({
      apiBaseUrl: 'http://api.test',
      apiKey: 'local-key',
      axiosConfig: {
        adapter: createAdapter({
          success: true,
          data: { model_id: 'model-1' },
          timestamp: '2026-04-28T00:00:00Z',
        }),
      },
    });

    await expect(client.get<{ model_id: string }>('/models/model-1')).resolves.toEqual({
      model_id: 'model-1',
    });
  });

  it('returns direct paginated payloads from list endpoints', async () => {
    const payload: PaginatedResponse<{ model_id: string }> = {
      items: [{ model_id: 'model-1' }],
      total: 1,
      page: 1,
      page_size: 20,
      pages: 1,
    };
    const client = new ApiClient({
      apiBaseUrl: 'http://api.test',
      apiKey: 'local-key',
      axiosConfig: { adapter: createAdapter(payload) },
    });

    await expect(client.get<PaginatedResponse<{ model_id: string }>>('/models')).resolves.toEqual(
      payload,
    );
  });

  it('raises structured errors for failed API responses', async () => {
    const client = new ApiClient({
      apiBaseUrl: 'http://api.test',
      apiKey: 'local-key',
      axiosConfig: {
        adapter: createAdapter(
          {
            success: false,
            error: 'Invalid API key.',
            error_code: 'UNAUTHORIZED',
            details: { header: 'X-API-Key' },
          },
          401,
        ),
      },
    });

    await expect(client.get('/models')).rejects.toMatchObject({
      message: 'Invalid API key.',
      statusCode: 401,
      errorCode: 'UNAUTHORIZED',
      details: { header: 'X-API-Key' },
    });
  });

  it('can reach the backend health endpoint outside the API prefix', async () => {
    const client = new ApiClient({
      apiBaseUrl: 'http://api.test',
      axiosConfig: {
        adapter: createAdapter({ status: 'healthy', timestamp: '2026-04-28T00:00:00Z' }),
      },
    });

    await expect(client.health()).resolves.toEqual({
      status: 'healthy',
      timestamp: '2026-04-28T00:00:00Z',
    });
  });
});