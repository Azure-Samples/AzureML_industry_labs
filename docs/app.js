// Global state
let labs = [];
let filteredLabs = [];
let selectedIndustries = new Set();
let selectedLanguages = new Set();
let selectedUseCases = new Set();

// GitHub repository info
const GITHUB_REPO = 'Azure-Samples/AzureML_industry_labs';
const GITHUB_BRANCH = 'main';

// Initialize the application
async function init() {
    try {
        const response = await fetch('labs-config.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        labs = await response.json();
        filteredLabs = [...labs];

        renderFilters();
        renderLabs();
        setupEventListeners();
    } catch (error) {
        console.error('Error loading labs configuration:', error);
        document.getElementById('labsGrid').innerHTML =
            '<p class="no-results">Error loading labs. Please check the configuration file.</p>';
    }
}

// Render filter checkboxes
function renderFilters() {
    const industries = new Set();
    const languages = new Set();
    const useCases = new Set();

    labs.forEach(lab => {
        if (lab.industry) industries.add(lab.industry);
        (lab.language || []).forEach(l => languages.add(l));
        (lab.useCase || []).forEach(u => useCases.add(u));
    });

    renderFilterGroup('industryFilters', Array.from(industries).sort(), 'industry');
    renderFilterGroup('languageFilters', Array.from(languages).sort(), 'language');
    renderFilterGroup('useCaseFilters', Array.from(useCases).sort(), 'useCase');
}

function renderFilterGroup(containerId, items, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = items.map(item => `
        <div class="filter-item">
            <input type="checkbox" id="${type}-${item.replace(/\s+/g, '-')}"
                   value="${item}" data-type="${type}">
            <label for="${type}-${item.replace(/\s+/g, '-')}">${item}</label>
        </div>
    `).join('');
}

// Render labs grid
function renderLabs() {
    const grid = document.getElementById('labsGrid');
    const noResults = document.getElementById('noResults');
    const labCount = document.getElementById('labCount');

    if (filteredLabs.length === 0) {
        grid.style.display = 'none';
        noResults.style.display = 'block';
        labCount.textContent = '';
        return;
    }

    grid.style.display = 'grid';
    noResults.style.display = 'none';
    labCount.textContent = `Showing ${filteredLabs.length} of ${labs.length} lab${labs.length !== 1 ? 's' : ''}`;

    grid.innerHTML = filteredLabs.map(lab => createLabCard(lab)).join('');

    // Add click handlers to cards
    document.querySelectorAll('.lab-card').forEach((card, index) => {
        card.addEventListener('click', () => showLabModal(filteredLabs[index]));
    });
}

function createLabCard(lab) {
    return `
        <div class="lab-card" data-lab-id="${escapeAttr(lab.directory)}">
            <div class="lab-content">
                <div class="lab-card-header">
                    <span class="tag industry">${escapeHtml(lab.industry)}</span>
                    ${lab.external ? '<span class="tag external">External</span>' : ''}
                </div>
                <h3 class="lab-title">${escapeHtml(lab.name)}</h3>
                <p class="lab-description">${escapeHtml(lab.shortDescription)}</p>
                <div class="lab-tags">
                    ${(lab.language || []).map(l =>
                        `<span class="tag language">${escapeHtml(l)}</span>`
                    ).join('')}
                    ${(lab.useCase || []).map(u =>
                        `<span class="tag use-case">${escapeHtml(u)}</span>`
                    ).join('')}
                </div>
                <div class="lab-footer">
                    <div class="lab-authors">
                        ${(lab.authors || []).map(author =>
                            `<a href="https://github.com/${encodeURIComponent(author)}" target="_blank" class="author-link" onclick="event.stopPropagation()">@${escapeHtml(author)}</a>`
                        ).join(', ')}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Show lab detail modal
function showLabModal(lab) {
    const modal = document.getElementById('labModal');
    const modalBody = document.getElementById('modalBody');

    modalBody.innerHTML = `
        <h2 class="modal-title">${escapeHtml(lab.name)}</h2>
        <div class="modal-meta">
            <div class="modal-meta-item">
                <h4>Industry</h4>
                <div class="lab-tags">
                    <span class="tag industry">${escapeHtml(lab.industry)}</span>
                    ${lab.external ? '<span class="tag external">External</span>' : ''}
                </div>
            </div>
            <div class="modal-meta-item">
                <h4>Language</h4>
                <div class="lab-tags">
                    ${(lab.language || []).map(l =>
                        `<span class="tag language">${escapeHtml(l)}</span>`
                    ).join('')}
                </div>
            </div>
            <div class="modal-meta-item">
                <h4>Use-Case</h4>
                <div class="lab-tags">
                    ${(lab.useCase || []).map(u =>
                        `<span class="tag use-case">${escapeHtml(u)}</span>`
                    ).join('')}
                </div>
            </div>
            <div class="modal-meta-item">
                <h4>Authors</h4>
                <div class="lab-authors">
                    ${(lab.authors || []).map(author =>
                        `<a href="https://github.com/${encodeURIComponent(author)}" target="_blank" class="author-link">@${escapeHtml(author)}</a>`
                    ).join(', ')}
                </div>
            </div>
        </div>
        <div class="modal-description">
            ${escapeHtml(lab.detailedDescription || lab.shortDescription)}
        </div>
        <div class="modal-actions">
            <a href="${escapeAttr(lab.githubPath || `https://github.com/${GITHUB_REPO}/tree/${GITHUB_BRANCH}/${lab.directory}`)}" class="btn-primary" target="_blank">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                View Lab on GitHub
            </a>
            ${lab.external ? '<p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.7;">This lab is hosted in an external repository.</p>' : ''}
        </div>
    `;

    modal.style.display = 'block';
}

// Filter labs based on search + checkboxes
function filterLabs() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();

    filteredLabs = labs.filter(lab => {
        // Search filter
        const matchesSearch = !searchTerm ||
            lab.name.toLowerCase().includes(searchTerm) ||
            lab.shortDescription.toLowerCase().includes(searchTerm) ||
            (lab.detailedDescription || '').toLowerCase().includes(searchTerm);

        // Industry filter
        const matchesIndustry = selectedIndustries.size === 0 ||
            selectedIndustries.has(lab.industry);

        // Language filter
        const matchesLanguage = selectedLanguages.size === 0 ||
            (lab.language || []).some(l => selectedLanguages.has(l));

        // Use-Case filter
        const matchesUseCase = selectedUseCases.size === 0 ||
            (lab.useCase || []).some(u => selectedUseCases.has(u));

        return matchesSearch && matchesIndustry && matchesLanguage && matchesUseCase;
    });

    renderLabs();
}

// Setup event listeners
function setupEventListeners() {
    // Search
    document.getElementById('searchInput').addEventListener('input', filterLabs);

    // Filter checkboxes
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const type = e.target.dataset.type;
            const value = e.target.value;
            const set = type === 'industry' ? selectedIndustries
                      : type === 'language' ? selectedLanguages
                      : selectedUseCases;

            if (e.target.checked) {
                set.add(value);
            } else {
                set.delete(value);
            }

            filterLabs();
        });
    });

    // Clear filters
    document.getElementById('clearFilters').addEventListener('click', () => {
        selectedIndustries.clear();
        selectedLanguages.clear();
        selectedUseCases.clear();
        document.getElementById('searchInput').value = '';
        document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
        filterLabs();
    });

    // Modal close
    const modal = document.getElementById('labModal');
    const closeBtn = document.querySelector('.modal-close');

    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
        }
    });
}

// Utility: escape HTML
function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str || ''));
    return div.innerHTML;
}

// Utility: escape attribute value
function escapeAttr(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
