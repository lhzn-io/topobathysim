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
        let html = `
            <div class="stats-row" style="margin-bottom:16px;">
                <span>Total Fused Cached Grid Cells: <strong>${d.count}</strong></span>
                <span style="margin-left:16px;">Total Size: <strong>${formatBytes(d.bytes)}</strong></span>
            </div>
        `;

        if (!d.policies || d.policies.length === 0) {
            return html + `<div class="detail-section"><em>No nested policy groups found.</em></div>`;
        }

        d.policies.forEach(p => {
            const zoomRows = Object.entries(p.by_zoom || {}).map(([z, i]) => {
                const resMeters = 156543.03 / Math.pow(2, z);
                const latDiff = Math.abs(i.lat_max - i.lat_min);
                const lonDiff = Math.abs(i.lon_max - i.lon_min);
                const avgLat = (i.lat_max + i.lat_min) / 2;
                const kmPerDegLat = 111.32;
                const kmPerDegLon = 111.32 * Math.cos(avgLat * Math.PI / 180);
                const areaSqKm = (latDiff * kmPerDegLat) * (lonDiff * kmPerDegLon);

                return `<tr>
                    <td><strong>L${z}</strong> <small style="color:#8b949e">~${resMeters.toFixed(1)}m</small></td>
                    <td>${i.count}</td>
                    <td>${formatBytes(i.bytes)}</td>
                    <td>${areaSqKm < 0.1 ? '<0.1' : areaSqKm.toFixed(1)} km²</td>
                    <td style="font-family:monospace; font-size:0.8rem">${i.lon_min?.toFixed(2)}, ${i.lat_min?.toFixed(2)}</td>
                </tr>`;
            }).join('');

            html += `
            <div class="detail-section" style="margin-top: 24px; padding: 12px; border: 1px solid #30363d; border-radius: 6px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#58a6ff;">Policy Hash: <span style="font-family:monospace; color:#c9d1d9;">${p.policy_hash}</span></h4>
                    <span style="font-size:0.9rem;">${p.count} grid cells · ${formatBytes(p.bytes)}</span>
                </div>
                `;

            if (p.yaml_snippet) {
                // Ensure safe HTML rendering of YAML snippet
                const safeYaml = p.yaml_snippet
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;");

                html += `
                <details style="margin-top: 12px; margin-bottom: 12px;">
                    <summary style="cursor:pointer; color:#8b949e; font-size:0.9rem;">View Policy YAML</summary>
                    <pre style="background:#161b22; padding:12px; border-radius:6px; overflow-x:auto; font-size:0.85rem; margin-top:8px;"><code>${safeYaml}</code></pre>
                </details>
                `;
            } else {
                html += `<div style="margin-top:12px; margin-bottom:12px; color:#8b949e; font-size:0.9rem;"><em>No YAML mapping found for this hash.</em></div>`;
            }

            html += `
                <table class="detail-table">
                    <thead><tr><th>Zoom / Res</th><th>Cached Grid Cells</th><th>Size</th><th>Coverage (Est)</th><th>Min Bounds</th></tr></thead>
                    <tbody>${zoomRows}</tbody>
                </table>
            </div>`;
        });

        return html;
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
    window.openPurgeModal = async (tierNums) => {
        if(!summaryData) {
             alert("Cache summary not loaded yet.");
             return;
        }

        if (!fullDetail) {
             const preloadBtn = document.getElementById('purgeGlobalBtn');
             if (preloadBtn) preloadBtn.disabled = true;
             await loadDetail(null, true);
             if (preloadBtn) preloadBtn.disabled = false;
        }

        window.selectedTier2Hashes = new Set(); // Reset sub-selections

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

        window.selectedTier2Hashes = window.selectedTier2Hashes || new Set();

        summaryData.tiers.forEach(t => {
            const row = document.createElement('div');
            row.className = 'purge-option-row';
            row.style.cssText = 'display:flex; align-items:center; padding:8px 0; gap:12px; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.9rem;';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = selectedPurgeTiers.has(t.number);
            checkbox.onchange = (e) => {
                if(e.target.checked) selectedPurgeTiers.add(t.number);
                else {
                    selectedPurgeTiers.delete(t.number);
                    if (t.number === 2) window.selectedTier2Hashes.clear();
                }
                renderPurgeModalList(); // Re-render to show/hide dynamic hash lists
                checkPurgeSafety();
            };

            const label = document.createElement('span');
            label.textContent = `Tier ${t.number}: ${t.name} (${formatBytes(t.bytes)})`;
            if (t.number === 5) label.style.color = '#ff7b72'; // Red for raw files

            row.appendChild(checkbox);
            row.appendChild(label);
            purgeTierListEl.appendChild(row);

            // Dynamically show the Tier 2 hashes if Tier 2 is selected
            if (t.number === 2 && selectedPurgeTiers.has(2) && typeof fullDetail !== 'undefined' && fullDetail && fullDetail.tier_2 && fullDetail.tier_2.policies && fullDetail.tier_2.policies.length > 0) {
                const subContainer = document.createElement('div');
                subContainer.style.cssText = 'padding-left: 32px; font-size: 0.85rem; color: #aaa; margin-bottom: 8px; margin-top: 4px;';

                const helpText = document.createElement('div');
                helpText.textContent = 'Select specific hashes to purge (leave all unchecked to purge the ENTIRE tier):';
                helpText.style.marginBottom = '6px';
                subContainer.appendChild(helpText);

                fullDetail.tier_2.policies.forEach(p => {
                    const hashRow = document.createElement('div');
                    hashRow.style.cssText = 'display:flex; align-items:center; gap:8px; padding: 2px 0;';

                    const hashCb = document.createElement('input');
                    hashCb.type = 'checkbox';
                    hashCb.checked = window.selectedTier2Hashes.has(p.policy_hash);
                    hashCb.onchange = (e) => {
                        if(e.target.checked) window.selectedTier2Hashes.add(p.policy_hash);
                        else window.selectedTier2Hashes.delete(p.policy_hash);
                    };

                    const hashLabel = document.createElement('span');
                    let snip = p.yaml_snippet || 'Custom Policy';
                    if (snip.length > 50) snip = snip.substring(0, 50) + '...';
                    hashLabel.textContent = `${p.policy_hash.substring(0,8)}... (${formatBytes(p.bytes)}) - ${snip}`;
                    hashLabel.title = p.yaml_snippet || '';

                    hashRow.appendChild(hashCb);
                    hashRow.appendChild(hashLabel);
                    subContainer.appendChild(hashRow);
                });
                purgeTierListEl.appendChild(subContainer);
            }
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
                    yes: true,
                    tier_2_hashes: (tiers.includes(2) && window.selectedTier2Hashes && window.selectedTier2Hashes.size > 0)
                                     ? Array.from(window.selectedTier2Hashes)
                                     : null
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
