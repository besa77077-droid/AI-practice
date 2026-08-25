(function () {
  "use strict";

  // ---- theme ----
  var THEME_KEY = "insight-engine-theme";
  var stored = localStorage.getItem(THEME_KEY);
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  var themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme");
      var isDark = current === "dark" || (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem(THEME_KEY, next);
    });
  }

  // ---- modals ----
  document.querySelectorAll("[data-open-modal]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var modal = document.getElementById(btn.getAttribute("data-open-modal"));
      if (modal) modal.hidden = false;
    });
  });
  document.querySelectorAll("[data-close-modal]").forEach(function (el) {
    el.addEventListener("click", function () {
      el.closest(".modal").hidden = true;
    });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal:not([hidden])").forEach(function (m) { m.hidden = true; });
    }
  });

  // ---- upload form: picking a file auto-selects its source radio ----
  document.querySelectorAll(".source-option input[type='file'], .source-option textarea").forEach(function (input) {
    input.addEventListener(input.type === "file" ? "change" : "input", function () {
      var radio = input.closest(".source-option").querySelector("input[type='radio']");
      if (radio) radio.checked = true;
    });
  });

  // ---- insight card collapse ----
  document.querySelectorAll("[data-toggle-card]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      btn.closest(".insight-card").classList.toggle("is-open");
    });
  });

  // ---- click-quote -> scroll + flash transcript segment (and back) ----
  function jumpToSegment(segId) {
    var line = document.getElementById("seg-" + segId);
    if (!line) return;
    line.scrollIntoView({ behavior: "smooth", block: "center" });
    line.classList.remove("flash-highlight");
    // restart animation even if already flashed once
    window.requestAnimationFrame(function () {
      line.classList.add("flash-highlight");
    });
  }

  document.querySelectorAll("[data-jump-to-segment]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      jumpToSegment(btn.getAttribute("data-jump-to-segment"));
      var card = btn.closest(".insight-card");
      if (card && !card.classList.contains("is-open")) card.classList.add("is-open");
    });
  });

  if (typeof window.__highlightSegment !== "undefined" && window.__highlightSegment !== null) {
    window.addEventListener("DOMContentLoaded", function () {
      jumpToSegment(window.__highlightSegment);
    });
  }

  // ---- processing page: poll job status ----
  var jobId = window.__jobId;
  if (jobId) {
    var STEP_ORDER = ["transcribe", "extract", "match", "cluster", "hypotheses"];
    var stepEls = {};
    document.querySelectorAll(".pipeline-step").forEach(function (el) {
      stepEls[el.getAttribute("data-step")] = el;
    });

    function applySteps(steps) {
      var seen = {};
      steps.forEach(function (s) { seen[s.name] = s.detail; });
      var reachedIdx = -1;
      STEP_ORDER.forEach(function (name, idx) {
        if (seen[name] !== undefined) reachedIdx = idx;
      });
      STEP_ORDER.forEach(function (name, idx) {
        var el = stepEls[name];
        if (!el) return;
        el.classList.remove("is-active", "is-done");
        if (idx < reachedIdx) {
          el.classList.add("is-done");
        } else if (idx === reachedIdx) {
          el.classList.add("is-active");
        }
        var detailEl = el.querySelector(".pipeline-step__detail");
        if (detailEl && seen[name] !== undefined) detailEl.textContent = seen[name];
      });
    }

    function poll() {
      fetch("/api/jobs/" + jobId)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          applySteps(data.steps || []);
          if (data.status === "done") {
            STEP_ORDER.forEach(function (name) {
              var el = stepEls[name];
              if (el) { el.classList.remove("is-active"); el.classList.add("is-done"); }
            });
            setTimeout(function () {
              window.location.href = "/interviews/" + data.interview_id;
            }, 500);
            return;
          }
          if (data.status === "error") {
            var errEl = document.getElementById("processing-error");
            if (errEl) {
              errEl.hidden = false;
              errEl.textContent = data.error || "Не удалось обработать интервью.";
            }
            return;
          }
          setTimeout(poll, 700);
        })
        .catch(function () {
          setTimeout(poll, 1500);
        });
    }
    poll();
  }
})();
