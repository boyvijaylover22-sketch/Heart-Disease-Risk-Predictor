document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('assessment-form');
    const resetButton = document.getElementById('reset-btn');
    const submitButton = document.querySelector('.btn-primary');

    if (submitButton && form) {
        form.addEventListener('submit', () => {
            submitButton.textContent = 'Analyzing...';
            submitButton.disabled = true;
        });
    }

    if (resetButton && form) {
        resetButton.addEventListener('click', () => {
            if (submitButton) {
                submitButton.textContent = 'Assess Risk';
                submitButton.disabled = false;
            }
        });
    }
});
