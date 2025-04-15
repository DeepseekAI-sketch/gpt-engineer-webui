/**
 * GPT-Engineer Web UI
 * Main JavaScript functionality
 */

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(message => {
        setTimeout(() => {
            const alert = new bootstrap.Alert(message);
            alert.close();
        }, 5000);
    });

    // Initialize code highlighting for any code blocks
    if (typeof hljs !== 'undefined') {
        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Initialize DataTables if present
    if (typeof $.fn.DataTable !== 'undefined') {
        $('.datatable').DataTable({
            responsive: true,
            pageLength: 10,
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search..."
            }
        });
    }

    // Collapsible sidebar toggler
    const sidebarToggle = document.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function(e) {
            e.preventDefault();
            document.body.classList.toggle('sidebar-collapsed');
            
            // Store state in localStorage
            if (document.body.classList.contains('sidebar-collapsed')) {
                localStorage.setItem('sidebar', 'collapsed');
            } else {
                localStorage.setItem('sidebar', 'expanded');
            }
        });
        
        // Restore sidebar state from localStorage
        if (localStorage.getItem('sidebar') === 'collapsed') {
            document.body.classList.add('sidebar-collapsed');
        }
    }

    // Project file copy buttons
    const copyButtons = document.querySelectorAll('.copy-btn');
    if (copyButtons.length > 0) {
        copyButtons.forEach(button => {
            button.addEventListener('click', function() {
                const targetId = this.dataset.clipboardTarget;
                const textToCopy = document.querySelector(targetId).textContent;
                
                navigator.clipboard.writeText(textToCopy).then(() => {
                    // Change button text temporarily
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="bi bi-check"></i> Copied!';
                    
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                });
            });
        });
    }

    // Temperature slider in project creation
    const temperatureSlider = document.getElementById('temperature');
    if (temperatureSlider) {
        const temperatureValue = document.getElementById('temperatureValue');
        temperatureSlider.addEventListener('input', function() {
            temperatureValue.textContent = this.value;
        });
    }

    // API key toggle visibility
    const toggleApiKey = document.getElementById('toggleApiKey');
    if (toggleApiKey) {
        toggleApiKey.addEventListener('click', function() {
            const apiKeyInput = document.getElementById('apiKey');
            const icon = this.querySelector('i');
            
            if (apiKeyInput.type === 'password') {
                apiKeyInput.type = 'text';
                icon.classList.remove('bi-eye');
                icon.classList.add('bi-eye-slash');
            } else {
                apiKeyInput.type = 'password';
                icon.classList.remove('bi-eye-slash');
                icon.classList.add('bi-eye');
            }
        });
    }

    // Project creation form validation
    const createProjectForm = document.querySelector('form[action*="create_project"]');
    if (createProjectForm) {
        createProjectForm.addEventListener('submit', function(event) {
            const projectName = document.getElementById('project_name').value;
            const prompt = document.getElementById('prompt').value;
            
            if (!projectName.trim()) {
                event.preventDefault();
                alert('Project name is required!');
                return false;
            }
            
            if (!prompt.trim()) {
                event.preventDefault();
                alert('Prompt is required!');
                return false;
            }
            
            return true;
        });
    }

    // Auto-refresh for project detail page when project is running
    const projectStatus = document.querySelector('[data-project-status="running"]');
    if (projectStatus) {
        const projectId = projectStatus.dataset.projectId;
        
        function updateProjectStatus() {
            fetch(`/api/projects/${projectId}/status`)
                .then(response => response.json())
                .then(data => {
                    // Update terminal output if available
                    console.log("Project status update:", data);  // Add this for debugging // Update UI...
                    const outputTerminal = document.getElementById('output-terminal');
                    if (outputTerminal && data.output && data.output.length > 0) {
                        outputTerminal.innerHTML = data.output.join('');
                        outputTerminal.scrollTop = outputTerminal.scrollHeight;
                    }
                    
                    // If job is still running, continue polling
                    if (data.status === 'running') {
                        setTimeout(updateProjectStatus, 3000);
                    } else {
                        // Reload page when job completes
                        window.location.reload();
                    }
                })
                .catch(error => {
                    console.error('Error updating project status:', error);
                    // Retry after a longer delay
                    setTimeout(updateProjectStatus, 5000);
                });
        }
        
        // Start polling
        setTimeout(updateProjectStatus, 3000);
    }
});
