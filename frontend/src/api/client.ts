import axios, {
  type AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
  type AxiosResponse,
} from 'axios';

const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';

/** Standard success envelope returned by most backend endpoints. */
export interface APIResponse<T> {
  success: boolean;
  data: T;
  message?: string | null;
  timestamp: string;
}

/** Standard error envelope returned by backend exception handlers. */
export interface APIErrorPayload {
  success: false;
  error: string;
  error_code: string;
  details?: Record<string, unknown> | null;
  timestamp?: string;
}

/** Generic paginated result returned by list endpoints. */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

/** Health-check payload returned by the backend root health endpoint. */
export interface HealthStatus {
  status: string;
  timestamp: string;
}

/** Constructor options for ApiClient. */
export interface ApiClientOptions {
  apiBaseUrl?: string;
  apiKey?: string;
  axiosConfig?: AxiosRequestConfig;
}

/** Error raised when the backend returns a structured API failure. */
export class ApiClientError extends Error {
  public readonly statusCode?: number;

  public readonly errorCode?: string;

  public readonly details?: Record<string, unknown>;

  public constructor(
    message: string,
    options: {
      statusCode?: number;
      errorCode?: string;
      details?: Record<string, unknown>;
      cause?: unknown;
    } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = 'ApiClientError';
    this.statusCode = options.statusCode;
    this.errorCode = options.errorCode;
    this.details = options.details;
    Object.setPrototypeOf(this, ApiClientError.prototype);
  }
}

const normalizeBaseUrl = (url: string): string => url.replace(/\/+$/, '');

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isWrappedResponse = <T>(value: APIResponse<T> | T): value is APIResponse<T> =>
  isRecord(value) && 'success' in value && 'data' in value;

const isErrorPayload = (value: unknown): value is APIErrorPayload =>
  isRecord(value) && value.success === false && typeof value.error === 'string';

const extractHeaders = (
  headers: AxiosRequestConfig['headers'],
): Record<string, string | number | boolean> => {
  if (!headers || !isRecord(headers)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(headers).filter(
      ([, value]) =>
        typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean',
    ),
  );
};

/** HTTP client for the Algo-Trading REST API. */
export class ApiClient {
  private readonly client: AxiosInstance;

  private readonly rootClient: AxiosInstance;

  public constructor(options: ApiClientOptions = {}) {
    const baseUrl = normalizeBaseUrl(options.apiBaseUrl || DEFAULT_API_BASE_URL);
    const apiKey = options.apiKey ?? DEFAULT_API_KEY;
    const configuredHeaders = extractHeaders(options.axiosConfig?.headers);
    const defaultHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (apiKey) {
      defaultHeaders['X-API-Key'] = apiKey;
    }

    const sharedConfig: AxiosRequestConfig = {
      ...options.axiosConfig,
      headers: {
        ...defaultHeaders,
        ...configuredHeaders,
      },
    };

    this.client = axios.create({
      ...sharedConfig,
      baseURL: `${baseUrl}/api/v1`,
    });
    this.rootClient = axios.create({
      ...sharedConfig,
      baseURL: baseUrl,
    });

    this.installInterceptors(this.client);
    this.installInterceptors(this.rootClient);
  }

  /** Return backend health state from the unauthenticated root endpoint. */
  public async health(): Promise<HealthStatus> {
    const response = await this.rootClient.get<HealthStatus>('/health');
    this.ensureSuccess(response);
    return response.data;
  }

  /** Execute an authenticated GET request against `/api/v1`. */
  public async get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
    const response = await this.client.get<APIResponse<T> | T>(url, { params });
    return this.handleResponse(response);
  }

  /** Execute an authenticated POST request against `/api/v1`. */
  public async post<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.post<APIResponse<T> | T>(url, data);
    return this.handleResponse(response);
  }

  /** Execute an authenticated PUT request against `/api/v1`. */
  public async put<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.put<APIResponse<T> | T>(url, data);
    return this.handleResponse(response);
  }

  /** Execute an authenticated DELETE request against `/api/v1`. */
  public async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete<APIResponse<T> | T>(url);
    return this.handleResponse(response);
  }

  private installInterceptors(client: AxiosInstance): void {
    client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: AxiosError<APIErrorPayload>) => {
        throw this.toApiClientError(error);
      },
    );
  }

  private unwrap<T>(payload: APIResponse<T> | T): T {
    if (isErrorPayload(payload)) {
      throw new ApiClientError(payload.error, {
        errorCode: payload.error_code,
        details: isRecord(payload.details) ? payload.details : undefined,
      });
    }

    if (isWrappedResponse(payload)) {
      if (!payload.success) {
        throw new ApiClientError('API response reported failure.');
      }

      return payload.data;
    }

    return payload;
  }

  private handleResponse<T>(response: AxiosResponse<APIResponse<T> | T>): T {
    this.ensureSuccess(response);
    return this.unwrap(response.data);
  }

  private ensureSuccess(response: AxiosResponse<unknown>): void {
    if (response.status < 400) {
      return;
    }

    const payload = response.data;

    if (isErrorPayload(payload)) {
      throw new ApiClientError(payload.error, {
        statusCode: response.status,
        errorCode: payload.error_code,
        details: isRecord(payload.details) ? payload.details : undefined,
      });
    }

    throw new ApiClientError(`API request failed with status ${response.status}.`, {
      statusCode: response.status,
    });
  }

  private toApiClientError(error: AxiosError<APIErrorPayload>): ApiClientError {
    const statusCode = error.response?.status;
    const payload = error.response?.data;

    if (isErrorPayload(payload)) {
      return new ApiClientError(payload.error, {
        statusCode,
        errorCode: payload.error_code,
        details: isRecord(payload.details) ? payload.details : undefined,
        cause: error,
      });
    }

    return new ApiClientError(error.message || 'Unable to reach the API.', {
      statusCode,
      cause: error,
    });
  }
}

export const apiClient = new ApiClient();