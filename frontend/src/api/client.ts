import axios, { AxiosError } from "axios";

const baseURL = import.meta.env.VITE_API_URL ?? "/api/v1";

export const api = axios.create({
  baseURL: baseURL.endsWith("/api/v1") ? baseURL : `${baseURL}/api/v1`,
  timeout: 60_000,
});

api.interceptors.response.use(
  (r) => r,
  (err: AxiosError<{ message?: string; error?: string }>) => {
    const msg =
      err.response?.data?.message ??
      err.response?.data?.error ??
      err.message ??
      "Request failed";
    return Promise.reject(new Error(msg));
  }
);
