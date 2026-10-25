import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RetryNotice from "./RetryNotice.vue";
import type { RetryView } from "../types";

function retry(overrides: Partial<RetryView> = {}): RetryView {
  return {
    scope: "turn",
    attempt: 2,
    maxRetries: 10,
    delayMs: 4000,
    nextAt: Date.now() + 4000,
    message: "Request failed (server)",
    ...overrides,
  };
}

describe("RetryNotice", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-25T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the attempt number and a one-second countdown", async () => {
    const wrapper = mount(RetryNotice, { props: { retry: retry() } });
    expect(wrapper.text()).toContain("第 2 次尝试");
    expect(wrapper.text()).toContain("4 秒后重试");

    vi.advanceTimersByTime(1000);
    await nextTick();
    expect(wrapper.text()).toContain("3 秒后重试");

    vi.advanceTimersByTime(3000);
    await nextTick();
    expect(wrapper.text()).toContain("正在重试");
    wrapper.unmount();
  });

  it("truncates the message and keeps the full text in the title", () => {
    const long = "x".repeat(100);
    const wrapper = mount(RetryNotice, {
      props: { retry: retry({ message: long }) },
    });
    expect(wrapper.text()).toContain(`${"x".repeat(80)}…`);
    expect(wrapper.text()).not.toContain(long);
    expect(wrapper.attributes("title")).toBe(long);
    wrapper.unmount();
  });

  it("labels summary retries separately", () => {
    const wrapper = mount(RetryNotice, {
      props: { retry: retry({ scope: "summary" }) },
    });
    expect(wrapper.text()).toContain("摘要压缩重试");
    wrapper.unmount();
  });
});
