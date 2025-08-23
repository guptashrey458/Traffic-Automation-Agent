#!/usr/bin/env node

/**
 * Dashboard Configuration Validator
 * Validates all dashboard components and configurations
 */

const fs = require('fs');
const path = require('path');

class DashboardValidator {
  constructor() {
    this.errors = [];
    this.warnings = [];
    this.validationResults = [];
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : '✅';
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  addError(message) {
    this.errors.push(message);
    this.log(message, 'error');
  }

  addWarning(message) {
    this.warnings.push(message);
    this.log(message, 'warning');
  }

  addSuccess(message) {
    this.log(message, 'success');
  }

  validateJsonFile(filePath) {
    try {
      if (!fs.existsSync(filePath)) {
        this.addError(`File not found: ${filePath}`);
        return null;
      }

      const content = fs.readFileSync(filePath, 'utf8');
      const json = JSON.parse(content);
      this.addSuccess(`Valid JSON: ${filePath}`);
      return json;
    } catch (error) {
      this.addError(`Invalid JSON in ${filePath}: ${error.message}`);
      return null;
    }
  }

  validateMainDashboard() {
    this.log('Validating main dashboard configuration...');
    
    const config = this.validateJsonFile('dashboard/main_dashboard.json');
    if (!config) return false;

    // Validate required sections
    const requiredSections = ['dashboard', 'configuration', 'layout', 'components'];
    for (const section of requiredSections) {
      if (!config[section]) {
        this.addError(`Missing required section: ${section}`);
      }
    }

    // Validate components
    if (config.components) {
      for (const component of config.components) {
        if (!component.id || !component.source || !component.position) {
          this.addError(`Invalid component configuration: ${JSON.stringify(component)}`);
        }
      }
      this.addSuccess(`Found ${config.components.length} components`);
    }

    // Validate layout
    if (config.layout) {
      if (config.layout.columns !== 12) {
        this.addWarning('Layout should use 12-column grid for optimal responsiveness');
      }
    }

    return true;
  }

  validateComponents() {
    this.log('Validating component configurations...');
    
    const componentDir = 'dashboard/components';
    if (!fs.existsSync(componentDir)) {
      this.addError(`Components directory not found: ${componentDir}`);
      return false;
    }

    const componentFiles = fs.readdirSync(componentDir).filter(f => f.endsWith('.json'));
    
    for (const file of componentFiles) {
      const filePath = path.join(componentDir, file);
      const config = this.validateJsonFile(filePath);
      
      if (config) {
        this.validateComponentStructure(config, file);
      }
    }

    this.addSuccess(`Validated ${componentFiles.length} component files`);
    return true;
  }

  validateComponentStructure(config, filename) {
    // Check required fields
    const requiredFields = ['component_type', 'name', 'position'];
    for (const field of requiredFields) {
      if (!config[field]) {
        this.addError(`Missing required field '${field}' in ${filename}`);
      }
    }

    // Validate position
    if (config.position) {
      const requiredPositionFields = ['x', 'y', 'width', 'height'];
      for (const field of requiredPositionFields) {
        if (typeof config.position[field] !== 'number') {
          this.addError(`Invalid position.${field} in ${filename}`);
        }
      }
    }

    // Component-specific validations
    switch (config.component_type) {
      case 'heatmap':
        this.validateHeatmapComponent(config, filename);
        break;
      case 'gantt_chart':
        this.validateGanttComponent(config, filename);
        break;
      case 'action_button':
        this.validateButtonComponent(config, filename);
        break;
      case 'modal':
        this.validateModalComponent(config, filename);
        break;
    }
  }

  validateHeatmapComponent(config, filename) {
    if (!config.data_source || !config.data_source.endpoint) {
      this.addError(`Heatmap missing data source endpoint in ${filename}`);
    }

    if (!config.config || !config.config.color_scale) {
      this.addError(`Heatmap missing color scale configuration in ${filename}`);
    }

    if (config.config && config.config.color_scale) {
      const colorScale = config.config.color_scale;
      if (!colorScale.colors || !Array.isArray(colorScale.colors)) {
        this.addError(`Invalid color scale colors in ${filename}`);
      }
      if (!colorScale.domain || !Array.isArray(colorScale.domain)) {
        this.addError(`Invalid color scale domain in ${filename}`);
      }
    }
  }

  validateGanttComponent(config, filename) {
    if (!config.data_source || !config.data_source.endpoint) {
      this.addError(`Gantt chart missing data source endpoint in ${filename}`);
    }

    if (!config.config || !config.config.time_axis) {
      this.addError(`Gantt chart missing time axis configuration in ${filename}`);
    }

    if (!config.config || !config.config.color_coding) {
      this.addError(`Gantt chart missing color coding configuration in ${filename}`);
    }
  }

  validateButtonComponent(config, filename) {
    if (!config.action || !config.action.type) {
      this.addError(`Action button missing action configuration in ${filename}`);
    }

    if (config.action && config.action.type === 'api_call') {
      if (!config.action.endpoint || !config.action.method) {
        this.addError(`API action missing endpoint or method in ${filename}`);
      }
    }
  }

  validateModalComponent(config, filename) {
    if (!config.content || !config.content.sections) {
      this.addError(`Modal missing content sections in ${filename}`);
    }

    if (!config.size && !config.config) {
      this.addWarning(`Modal missing size configuration in ${filename}`);
    }
  }

  validateModals() {
    this.log('Validating modal configurations...');
    
    const modalDir = 'dashboard/modals';
    if (!fs.existsSync(modalDir)) {
      this.addWarning(`Modals directory not found: ${modalDir}`);
      return true;
    }

    const modalFiles = fs.readdirSync(modalDir).filter(f => f.endsWith('.json'));
    
    for (const file of modalFiles) {
      const filePath = path.join(modalDir, file);
      const config = this.validateJsonFile(filePath);
      
      if (config) {
        this.validateModalComponent(config, file);
      }
    }

    this.addSuccess(`Validated ${modalFiles.length} modal files`);
    return true;
  }

  validateApiEndpoints() {
    this.log('Validating API endpoint references...');
    
    const endpoints = new Set();
    
    // Collect all endpoint references
    this.collectEndpointsFromFile('dashboard/main_dashboard.json', endpoints);
    this.collectEndpointsFromDirectory('dashboard/components', endpoints);
    this.collectEndpointsFromDirectory('dashboard/modals', endpoints);

    // Validate against expected API structure
    const expectedEndpoints = [
      '/flights/peaks',
      '/flights/risks', 
      '/optimize',
      '/whatif',
      '/constraints',
      '/airports',
      '/status'
    ];

    for (const endpoint of expectedEndpoints) {
      if (!endpoints.has(endpoint)) {
        this.addWarning(`Expected API endpoint not referenced: ${endpoint}`);
      }
    }

    for (const endpoint of endpoints) {
      if (!expectedEndpoints.includes(endpoint)) {
        this.addWarning(`Unknown API endpoint referenced: ${endpoint}`);
      }
    }

    this.addSuccess(`Found ${endpoints.size} API endpoint references`);
    return true;
  }

  collectEndpointsFromFile(filePath, endpoints) {
    if (!fs.existsSync(filePath)) return;
    
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const matches = content.match(/"endpoint":\s*"([^"]+)"/g);
      
      if (matches) {
        for (const match of matches) {
          const endpoint = match.match(/"endpoint":\s*"([^"]+)"/)[1];
          endpoints.add(endpoint);
        }
      }
    } catch (error) {
      this.addWarning(`Error reading ${filePath}: ${error.message}`);
    }
  }

  collectEndpointsFromDirectory(dirPath, endpoints) {
    if (!fs.existsSync(dirPath)) return;
    
    const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.json'));
    
    for (const file of files) {
      this.collectEndpointsFromFile(path.join(dirPath, file), endpoints);
    }
  }

  validateResponsiveness() {
    this.log('Validating responsive design configuration...');
    
    const config = this.validateJsonFile('dashboard/main_dashboard.json');
    if (!config) return false;

    // Check responsive breakpoints
    if (!config.responsive_breakpoints) {
      this.addError('Missing responsive breakpoints configuration');
      return false;
    }

    const breakpoints = config.responsive_breakpoints;
    const requiredBreakpoints = ['mobile', 'tablet', 'desktop'];
    
    for (const bp of requiredBreakpoints) {
      if (typeof breakpoints[bp] !== 'number') {
        this.addError(`Invalid or missing breakpoint: ${bp}`);
      }
    }

    // Check mobile layout
    if (!config.mobile_layout) {
      this.addWarning('Missing mobile layout configuration');
    } else {
      if (!config.mobile_layout.component_overrides) {
        this.addWarning('Mobile layout missing component overrides');
      }
    }

    this.addSuccess('Responsive design configuration validated');
    return true;
  }

  validateAccessibility() {
    this.log('Validating accessibility features...');
    
    // Check for tooltip system
    const tooltipConfig = this.validateJsonFile('dashboard/components/tooltips_glossary.json');
    if (!tooltipConfig) {
      this.addWarning('Missing tooltip system for accessibility');
    } else {
      if (!tooltipConfig.glossary || Object.keys(tooltipConfig.glossary).length === 0) {
        this.addError('Tooltip glossary is empty');
      } else {
        this.addSuccess(`Found ${Object.keys(tooltipConfig.glossary).length} tooltip definitions`);
      }
    }

    // Check for help system
    const helpConfig = this.validateJsonFile('dashboard/modals/help_modal.json');
    if (!helpConfig) {
      this.addWarning('Missing help system');
    } else {
      this.addSuccess('Help system configured');
    }

    return true;
  }

  validatePerformance() {
    this.log('Validating performance configurations...');
    
    const config = this.validateJsonFile('dashboard/main_dashboard.json');
    if (!config) return false;

    // Check refresh intervals
    if (config.configuration) {
      const refreshInterval = config.configuration.refresh_interval;
      if (!refreshInterval || refreshInterval < 10000) {
        this.addWarning('Refresh interval should be at least 10 seconds to avoid excessive API calls');
      }
    }

    // Check for caching configurations
    let cachingFound = false;
    this.checkCachingInDirectory('dashboard/components', (found) => {
      if (found) cachingFound = true;
    });

    if (!cachingFound) {
      this.addWarning('Consider implementing caching for better performance');
    }

    this.addSuccess('Performance configuration reviewed');
    return true;
  }

  checkCachingInDirectory(dirPath, callback) {
    if (!fs.existsSync(dirPath)) return;
    
    const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.json'));
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(path.join(dirPath, file), 'utf8');
        if (content.includes('cache') || content.includes('Cache')) {
          callback(true);
          return;
        }
      } catch (error) {
        // Ignore read errors
      }
    }
    
    callback(false);
  }

  runAllValidations() {
    console.log('🚀 Starting Dashboard Validation...\n');

    const validations = [
      { name: 'Main Dashboard', fn: () => this.validateMainDashboard() },
      { name: 'Components', fn: () => this.validateComponents() },
      { name: 'Modals', fn: () => this.validateModals() },
      { name: 'API Endpoints', fn: () => this.validateApiEndpoints() },
      { name: 'Responsiveness', fn: () => this.validateResponsiveness() },
      { name: 'Accessibility', fn: () => this.validateAccessibility() },
      { name: 'Performance', fn: () => this.validatePerformance() }
    ];

    for (const validation of validations) {
      console.log(`\n📋 Validating ${validation.name}...`);
      try {
        validation.fn();
      } catch (error) {
        this.addError(`Validation failed for ${validation.name}: ${error.message}`);
      }
    }

    this.printSummary();
  }

  printSummary() {
    console.log('\n📊 Validation Summary');
    console.log('====================');
    
    const totalIssues = this.errors.length + this.warnings.length;
    
    if (this.errors.length === 0 && this.warnings.length === 0) {
      console.log('🎉 All validations passed! Dashboard is ready for deployment.');
    } else {
      console.log(`❌ Errors: ${this.errors.length}`);
      console.log(`⚠️  Warnings: ${this.warnings.length}`);
      console.log(`📊 Total Issues: ${totalIssues}`);
      
      if (this.errors.length > 0) {
        console.log('\n❌ Critical Errors (must fix):');
        this.errors.forEach((error, i) => console.log(`  ${i + 1}. ${error}`));
      }
      
      if (this.warnings.length > 0) {
        console.log('\n⚠️  Warnings (recommended fixes):');
        this.warnings.forEach((warning, i) => console.log(`  ${i + 1}. ${warning}`));
      }
    }
    
    console.log('\n✅ Dashboard validation complete!');
    
    // Exit with appropriate code
    process.exit(this.errors.length > 0 ? 1 : 0);
  }
}

// Run validation if executed directly
if (require.main === module) {
  const validator = new DashboardValidator();
  validator.runAllValidations();
}

module.exports = DashboardValidator;