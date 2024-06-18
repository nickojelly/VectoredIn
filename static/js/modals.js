// Get the modal elements
const axisModal = document.getElementById('axis-modal');
const correlationsModal = document.getElementById('correlations-modal');

// Get the close button elements
const closeAxisModalBtn = document.getElementById('close-axis-modal');
const closeCorrelationsModalBtn = document.getElementById('close-correlations-modal');

// Get the elements that trigger the modals (e.g., tooltip icons)
const axisTooltipIcon = document.querySelector('.axis-tooltip-icon');
const correlationsTooltipIcon = document.querySelector('.correlations-tooltip-icon');

// Function to open the axis modal
function openAxisModal() {
  axisModal.classList.add('is-active');
}

// Function to open the correlations modal
function openCorrelationsModal() {
  correlationsModal.classList.add('is-active');
}

// Function to close the modals
function closeModal(modal) {
  modal.classList.remove('is-active');
}

// Add event listeners to the tooltip icons only if the elements exist
if (axisTooltipIcon) {
  axisTooltipIcon.addEventListener('click', openAxisModal);
} else {
  console.log('axisTooltipIcon element not found');
}

if (correlationsTooltipIcon) {
  correlationsTooltipIcon.addEventListener('click', openCorrelationsModal);
} else {
  console.log('correlationsTooltipIcon element not found');
}

// Add event listeners to the close buttons
closeAxisModalBtn.addEventListener('click', () => closeModal(axisModal));
closeCorrelationsModalBtn.addEventListener('click', () => closeModal(correlationsModal));

// Add event listeners to the modal backgrounds
axisModal.querySelector('.modal-background').addEventListener('click', () => closeModal(axisModal));
correlationsModal.querySelector('.modal-background').addEventListener('click', () => closeModal(correlationsModal));