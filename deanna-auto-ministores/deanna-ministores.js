(function () {
    function qs(root, sel) {
      return root.querySelector(sel);
    }
  
    async function handleClick(btn) {
      const root = btn.closest(".deanna-ministores-ondemand");
      const status = qs(root, ".deanna-ministores-status");
      const output = qs(root, ".deanna-ministores-output");
      const postId = btn.getAttribute("data-post-id");
  
      btn.disabled = true;
      status.textContent = "Generando...";
      output.innerHTML = "";
  
      const form = new FormData();
      form.append("action", "deanna_ministores_generate");
      form.append("nonce", DeannaMinistores.nonce);
      form.append("post_id", postId);
  
      try {
        const res = await fetch(DeannaMinistores.ajaxUrl, { method: "POST", body: form });
        const data = await res.json();
  
        if (!data || !data.success) {
          const msg = (data && data.data && data.data.message) ? data.data.message : "Error desconocido";
          status.textContent = "Error";
          output.innerHTML = `<div style="background:#111827;color:#f9fafb;padding:12px;border-radius:10px;">${msg}</div>`;
        } else {
          status.textContent = data.data.cached ? "Listo (cache)" : "Listo";
          output.innerHTML = data.data.html || "";
        }
      } catch (e) {
        status.textContent = "Error";
        output.innerHTML = `<div style="background:#111827;color:#f9fafb;padding:12px;border-radius:10px;">${String(e)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }
  
    document.addEventListener("click", function (e) {
      const btn = e.target.closest(".deanna-ministores-btn");
      if (!btn) return;
      e.preventDefault();
      handleClick(btn);
    });
  })();
  