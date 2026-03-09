document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Matching IDs from cache-viewer.html
    const tierListEl = document.getElementById('tierList');
    const detailContentEl = document.getElementById('detailContent');
    const detailTitleEl = document.getElementById('detailTitle');
    const totalSizeEl = document.getElementById('totalSize');
    const lastUpdatedEl = document.getElementById('lastUpdated');
    const cacheRootBadge = document.getElementById('cacheRootPath');
    const refreshBtn = document.getElementById('refreshBtn');

    // Purge Modal Elements
    const purgeModal = document.getElementById('purgeModal');
    // For modal close, we use class selector as there are multiple close buttons
    const closeButtons = document.querySelectorAll('.close-btn');

    // Purge Elements inside modal
    const confirmPurgeBtn = document.getElementById('confirmPurgeBtn');
    const purgeConfirmInput = document.getElementById('purgeConfirmInput');
    const dryRunToggle = document.getElementById('dryRunToggle');
    const purgeTierListEl = document.getElementById('purgeTierList');
    const purgeWarningEl = document.getElementById('purgeWarning');
    const purgeLogEl = document.getElementById('purgeLog');
    const purgeResultsEl = document.getElementById('purgeResults');

    // Global Purge Button (Header)
    const openGlobalPurgeBtn = document.getElementById('purgeBtn');

    // State
    let currentTierNum = null;
    let fullDetail = null;
    let summaryData = null;
    let selectedPurgeTiers = new Set();

    // --- Helpers ---

    function formatBytes(bytes, decimals = 2) {
        if (!bytes && bytes !== 0) return '-';
        if (bytes === 0) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function formatDate(ts) {
        if (!ts) return 'Never';
        // Python time.time() is seconds, JS Date is ms
        return new Date(ts * 1000).toLocaleString();
    }

    // --- Core Logic ---

    async function loadSummary() {
        tierListEl.innerHTML = '<div class="loading-state" style="padding:20px; text-align:center;"><div class="spinner" style="border: 2px solid rgba(255,255,255,0.1); border-top: 2px solid #58a6ff; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; margin:0 auto;"></div></div>';
        try {
            const response = await fetch('/api/v1/cache/summary?cache_bust=true');
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server Error (${response.status}): ${errText}`);
            }
            summaryData = await response.json();

            // Render Header Stats
            if(totalSizeEl) totalSizeEl.textContent = formatBytes(summaryData.total_bytes || 0);
            if(cacheRootBadge) cacheRootBadge.textContent = summaryData.cache_root || 'Unknown';
            // lastUpdated is a timestamp float
            if(lastUpdatedEl) lastUpdatedEl.textContent = formatDate(summaryData.last_updated);

            renderTierList(summaryData.tiers || []);

            // If we have a selection, load details roughly at same time
            // Otherwise wait for user interaction
            if (currentTierNum) {
                loadDetail(currentTierNum);
            } else {
                 // Pre-fetch details in background so click is fast
                 loadDetail(null, true);
            }

        } catch (err) {
            console.error(err);
            tierListEl.innerHTML = `<div style="padding:16px; color:#ff7b72; font-size:0.9rem;">Failed to load summary.<br><small>${err.message}</small></div>`;
        }
    }

    async function loadDetail(tierNum, background = false) {
        if (!background && tierNum) {
            detailContentEl.innerHTML = '<div class="loading-state" style="padding:40px; text-align:center; opacity:0.7;">Loading details...</div>';
        }

        try {
            const response = await fetch('/api/v1/cache/detail?cache_bust=false'); // use cached if valid (server handles Busting logic on summary usually)
            if (!response.ok) {
                // If detail fails (e.g. 500), we probably can't render much
                if(!background) throw new Error("Failed to load details");
                return;
            }
            fullDetail = await response.json();

            if (tierNum) {
                renderDetail(tierNum);
            }
        } catch (err) {
             console.error("Detail load failed", err);
             if (!background && tierNum) {
                 detailContentEl.innerHTML = `<div style="padding:20px; color:#ff7b72;">Failed to load detail view: ${err.message}</div>`;
             }
        }
    }

    // --- Rendering ---

    function renderTierList(tiers) {
        tierListEl.innerHTML = '';

        tiers.sort((a, b) => a.number - b.number).forEach(tier => {
            const card = document.createElement('div');
            card.className = `tier-card ${currentTierNum === tier.number ? 'selected' : ''}`;
            card.dataset.id = tier.number;
            card.onclick = () => selectTier(tier.number);

            const warningHtml = tier.warning ?
                `<span class="tier-badge" style="background:rgba(210,153,34,0.2); color:#d29922; margin-left:8px;" title="${tier.purge_reason}">⚠️ Warning</span>` : '';

            card.innerHTML = `
                <div class="tier-header">
                    <span class="tier-title">Tier ${tier.number}</span>
                    <span class="tier-badge">${(tier.items || 0).toLocaleString()} items</span>
                </div>
                <div class="tier-desc" style="font-size:0.85rem; color:#8b949e; margin-bottom:6px;">${tier.name}</div>
                <div class="tier-meta">
                    <span style="font-family:monospace;">${formatBytes(tier.bytes)}</span>
                    ${warningHtml}
                </div>
            `;
            tierListEl.appendChild(card);
        });
    }

    function selectTier(tierNum) {
        currentTierNum = tierNum;
        // Update selection UI
        document.querySelectorAll('.tier-card').forEach(c => {
            c.classList.toggle('selected', parseInt(c.dataset.id) === tierNum);
        });

        if (fullDetail) {
            renderDetail(tierNum);
        } else {
            loadDetail(tierNum);
        }
    }

    function renderDetail(tierNum) {
        if (!fullDetail) return;

        const detailKey = `tier_${tierNum}`;
        const data = fullDetail[detailKey];

        // Update Title
        const tierInfo = summaryData ? summaryData.tiers.find(t => t.number === tierNum) : { name: `Tier ${tierNum}` };
        if(detailTitleEl) detailTitleEl.textContent = `${tierInfo ? tierInfo.name : 'Unknown Tier'} Details`;

        if (!data) {
            detailContentEl.innerHTML = `<div style="padding:20px;">No detailed data found for Tier ${tierNum}</div>`;
            return;
        }

        let content = '';

        // Render helper based on tier
        if (tierNum === 1) content = renderTier1(data);
        else if (tierNum === 2) content = renderTier2(data);
        else if (tierNum === 3) content = renderTier3(data);
        else if (tierNum === 4) content = renderTier4(data);
        else if (tierNum === 5) content = renderTier5(data);

        // Add Purge Button specific to this tier
        const purgeBtnHtml = `
            <div style="margin-top:32px; border-top:1px solid rgba(255,255,255,0.1); padding-top:16px; display:flex; justify-content:flex-end;">
                <button class="btn btn-danger" onclick="window.openPurgeModal([${tierNum}])">
                    Purge Tier ${tierNum} Only...
                </button>
            </div>
        `;

        detailContentEl.innerHTML = content + purgeBtnHtml;
    }

    // --- Sub-Renderers ---
    function renderTier1(d) { // Tiles
        const rows = d.tile_subdirs.map(s => `<tr><td>${s.label}</td><td>${s.files}</td><td>${formatBytes(s.bytes)}</td></tr>`).join('');
        return `
            <div class="detail-section"><h4>Subdirectories</h4>
            <table class="detail-table"><thead><tr><th>Name</th><th>Files</th><th>Size</th></tr></thead><tbody>${rows}</tbody></table></div>
            <div class="detail-section"><h4>Metadata</h4><p>Meta JSONs: ${d.meta_count}</p><p>NPZ Arrays: ${d.npz_count}</p></div>`;
    }

    function renderTier2(d) { // Fused
        const zoomRows = Object.entries(d.by_zoom || {}).map(([z, i]) =>
            `<tr><td>L${z}</td><td>${i.count}</td><td>${formatBytes(i.bytes)}</td><td>${i.lon_min?.toFixed(2)}, ${i.lat_min?.toFixed(2)}</td></tr>`
        ).join('');
        return `
            <div class="stats-row" style="margin-bottom:16px;">
                <span>chunks fused: <strong>${d.count}</strong></span>
            </div>
            <div class="detail-section"><h4>Zoom Levels</h4>
            <table class="detail-table"><thead><tr><th>Zoom</th><th>Count</th><th>Size</th><th>Bounds</th></tr></thead><tbody>${zoomRows}</tbody></table></div>`;
    }

    function renderTier3(d) { // Providers
        const rows = d.providers.map(p => `<tr><td>${p.name}</td><td>${p.count}</td><td>${formatBytes(p.bytes)}</td><td>${formatDate(p.newest)}</td></tr>`).join('');
        return `<div class="detail-section"><h4>Providers</h4><table class="detail-table"><thead><tr><th>Name</th><th>Count</th><th>Size</th><th>Newest</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }

    function renderTier4(d) { // Discovery
        const rows = d.discovery.map(i => `<tr><td>${i.label}</td><td>${i.ok?'OK':'Err'}</td><td>${i.entries}</td><td>${formatBytes(i.bytes)}</td></tr>`).join('');
        return `<div class="detail-section"><h4>Metadata Files</h4><table class="detail-table"><thead><tr><th>File</th><th>Status</th><th>Entries</th><th>Size</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }

    function renderTier5(d) { // Raw
        const rows = d.raw_files.map(r => `<tr><td>${r.provider}</td><td>${r.count}</td><td>${formatBytes(r.bytes)}</td><td>${formatDate(r.newest)}</td></tr>`).join('');
        return `<div class="detail-section"><h4>Raw Source Files</h4><p style="color:#faa; font-size:0.9rem; margin-bottom:8px;">Warning: Deleting these requires re-download.</p><table class="detail-table"><thead><tr><th>Provider</th><th>Count</th><th>Size</th><th>Newest</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }


    // --- Purge Logic ---

    // Expose globally for inline onclick
    window.openPurgeModal = (tierNums) => {
        if(!summaryData) {
             alert("Cache summary not loaded yet.");
             return;
        }

        // If specific tiers requested (array), select them
        // If empty array passed (global button), select none initially (or maybe Tier 1?)
        selectedPurgeTiers.clear();
        tierNums.forEach(n => selectedPurgeTiers.add(n));

        renderPurgeModalList();

        if(purgeModal) purgeModal.classList.remove('hidden');
        if(purgeResultsEl) purgeResultsEl.classList.add('hidden');
        if(purgeLogEl) purgeLogEl.textContent = '';

        // Reset inputs
        if(purgeConfirmInput) purgeConfirmInput.value = '';
        checkPurgeSafety();
    };

    function renderPurgeModalList() {
        if (!summaryData || !purgeTierListEl) return;
        purgeTierListEl.innerHTML = '';

        summaryData.tiers.forEach(t => {
            const row = document.createElement('div');
            row.className = 'purge-option-row';
            row.style.cssText = 'display:flex; align-items:center; padding:8px 0; gap:12px; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.9rem;';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = selectedPurgeTiers.has(t.number);
            checkbox.onchange = (e) => {
                if(e.target.checked) selectedPurgeTiers.add(t.number);
                else selectedPurgeTiers.delete(t.number);
                checkPurgeSafety();
            };

            const label = document.createElement('span');
            label.textContent = `Tier ${t.number}: ${t.name} (${formatBytes(t.bytes)})`;
            if (t.number === 5) label.style.color = '#ff7b72'; // Red for raw files

            row.appendChild(checkbox);
            row.appendChild(label);
            purgeTierListEl.appendChild(row);
        });
    }

    function checkPurgeSafety() {
        if(!confirmPurgeBtn) return;

        const hasTier5 = selectedPurgeTiers.has(5);
        const count = selectedPurgeTiers.size;

        if (hasTier5) {
            purgeWarningEl.classList.remove('hidden');
            const isConfirmed = purgeConfirmInput.value === 'PURGE';
            confirmPurgeBtn.disabled = !isConfirmed;
        } else {
            purgeWarningEl.classList.add('hidden');
            confirmPurgeBtn.disabled = count === 0;
            // Also enable if > 0
        }
    }

    // Event Listeners
    if(openGlobalPurgeBtn) openGlobalPurgeBtn.onclick = () => window.openPurgeModal([]); // Open empty selection

    if(closeButtons) closeButtons.forEach(btn => btn.onclick = () => purgeModal.classList.add('hidden'));

    if(purgeConfirmInput) purgeConfirmInput.oninput = checkPurgeSafety;

    if(confirmPurgeBtn) confirmPurgeBtn.onclick = async () => {
        const isDryRun = dryRunToggle ? dryRunToggle.checked : true;
        const tiers = Array.from(selectedPurgeTiers);

        confirmPurgeBtn.disabled = true;
        confirmPurgeBtn.textContent = isDryRun ? 'Running Check...' : 'Purging...';

        if(purgeResultsEl) purgeResultsEl.classList.remove('hidden');
        if(purgeLogEl) purgeLogEl.textContent = 'Requesting...';

        try {
            const res = await fetch('/api/v1/cache/purge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    tiers: tiers,
                    dry_run: isDryRun,
                    yes: true
                })
            });

            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || 'Purge failed');

            // Show results
            let log = `Status: ${data.status}\nMessage: ${data.message}\n\n`;
            if (data.summary) {
                Object.entries(data.summary).forEach(([k, v]) => {
                    log += `${k}:\n  deleted ${v.count} files\n  freed ${formatBytes(v.bytes)}\n`;
                });
            } else {
                log += "No summary data returned.";
            }

            if (!isDryRun && data.status === 'success') {
                log += "\nRefetching summary...";
                loadSummary();
            }
            purgeLogEl.textContent = log;

        } catch (err) {
            purgeLogEl.textContent = `Error: ${err.message}`;
        } finally {
            confirmPurgeBtn.disabled = false;
            confirmPurgeBtn.textContent = 'Run Purge';
        }
    };

    if(refreshBtn) refreshBtn.onclick = loadSummary;

    // Integrity/Orphan Buttons (Sidebar placeholders)
    const checkBtn = document.querySelector('a[data-view="check"]');
    if(checkBtn) checkBtn.onclick = (e) => {
        e.preventDefault();
        alert("Integrity Check running in background... check server logs.");
        fetch('/api/v1/cache/check', { method: 'POST' });
    };

    const orphanBtn = document.querySelector('a[data-view="orphans"]');
    if(orphanBtn) orphanBtn.onclick = (e) => {
        e.preventDefault();
        alert("Orphan cleanup not yet implemented via UI.");
    };

    // Initial Start
    loadSummary();
});

// Add keyframes for spinner programmatically
const style = document.createElement('style');
style.textContent = `
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
`;
document.head.appendChild(style);
