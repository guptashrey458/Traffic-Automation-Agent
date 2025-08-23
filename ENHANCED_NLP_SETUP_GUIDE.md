# Enhanced Multi-Provider NLP Interface Setup Guide

## Overview

The Enhanced NLP Interface uses multiple AI providers in a fallback chain to ensure 100% availability and optimal performance. This guide will help you configure the API keys for all providers.

## Provider Fallback Chain

1. **Gemini (Primary)** - Google's Gemini 1.5 Flash-8B
2. **Perplexity (Secondary)** - Llama-3.1-Sonar model
3. **OpenAI (Tertiary)** - GPT-3.5-turbo
4. **Local (Fallback)** - Pattern-based matching (always available)

## API Key Setup

### 1. Google Gemini API Key (Recommended)

**Why:** Primary provider with excellent performance and generous free tier.

**Steps:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Add to your `.env` file:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

**Free Tier:** 15 requests per minute, 1,500 requests per day

### 2. Perplexity API Key (Optional but Recommended)

**Why:** Excellent secondary provider with good performance and reliability.

**Steps:**
1. Visit [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. Sign up for a Perplexity account
3. Generate an API key
4. Add to your `.env` file:
   ```bash
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```

**Pricing:** Pay-per-use, typically $0.001-0.002 per request

### 3. OpenAI API Key (Optional)

**Why:** Reliable tertiary provider, good for backup when other providers fail.

**Steps:**
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign up for an OpenAI account
3. Add billing information (required for API access)
4. Create a new API key
5. Add to your `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

**Pricing:** GPT-3.5-turbo: $0.0015 per 1K input tokens, $0.002 per 1K output tokens

## Configuration Options

Add these settings to your `.env` file to customize the enhanced NL interface:

```bash
# Enable the enhanced multi-provider NL interface
ENHANCED_NL_ENABLED=true

# Primary NLP provider (gemini, perplexity, openai, local)
ENHANCED_NL_PRIMARY_PROVIDER=gemini

# Enable autonomous actions (requires user permissions)
ENHANCED_NL_ENABLE_AUTONOMOUS_ACTIONS=true

# Minimum confidence threshold for autonomous actions (0.0-1.0)
ENHANCED_NL_CONFIDENCE_THRESHOLD=0.7

# Cache TTL for NLP responses (in seconds)
ENHANCED_NL_CACHE_TTL_SECONDS=300
```

## Minimum Setup (Free)

For a completely free setup, you only need:

```bash
# Only Gemini API key required (free tier)
GEMINI_API_KEY=your_gemini_api_key_here
ENHANCED_NL_ENABLED=true
```

The system will automatically fall back to the local provider if Gemini fails or hits rate limits.

## Recommended Setup (Best Performance)

For optimal performance and reliability:

```bash
# All three providers for maximum reliability
GEMINI_API_KEY=your_gemini_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Enhanced NL configuration
ENHANCED_NL_ENABLED=true
ENHANCED_NL_PRIMARY_PROVIDER=gemini
ENHANCED_NL_ENABLE_AUTONOMOUS_ACTIONS=true
ENHANCED_NL_CONFIDENCE_THRESHOLD=0.7
ENHANCED_NL_CACHE_TTL_SECONDS=300
```

## Testing Your Setup

After configuring your API keys, test the setup:

```bash
# Run the enhanced NL interface demo
python demo_enhanced_nl_interface.py

# Run the tests to verify functionality
python -m pytest tests/test_enhanced_nl_interface.py -v
```

## Provider Selection Strategy

The system automatically selects providers based on:

1. **Availability**: Checks if API key is configured
2. **Rate Limits**: Respects provider rate limits
3. **Performance**: Tracks success rates and response times
4. **Cost**: Uses free/cheaper providers first when possible

## Troubleshooting

### Common Issues:

1. **"Provider not available" errors**
   - Check that API keys are correctly set in `.env`
   - Verify API keys are valid and not expired
   - Ensure you have sufficient quota/credits

2. **Rate limit errors**
   - The system automatically handles rate limits with fallback
   - Consider upgrading to paid tiers for higher limits
   - Check provider usage dashboards

3. **All providers failing**
   - The local provider should always work as fallback
   - Check network connectivity
   - Verify `.env` file is being loaded correctly

### Debug Mode:

Enable debug logging to see provider selection:

```bash
LOGGING__LEVEL=DEBUG
```

This will show which providers are being tried and why they succeed/fail.

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables only**
3. **Rotate API keys regularly**
4. **Monitor API usage for unexpected spikes**
5. **Set up billing alerts for paid providers**

## Cost Optimization

1. **Start with Gemini free tier** (most generous)
2. **Enable caching** to reduce API calls
3. **Use local fallback** for simple queries
4. **Monitor usage** through provider dashboards
5. **Set up usage alerts** to avoid unexpected charges

## Support

If you encounter issues:

1. Check the logs for detailed error messages
2. Run the test suite to identify specific problems
3. Verify API keys are valid and have sufficient quota
4. Check provider status pages for outages

The enhanced NL interface is designed to be resilient - even if all external providers fail, the local provider ensures the system remains functional.