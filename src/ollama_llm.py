import json
import re
import logging
from ollama import chat
from ollama import ChatResponse
from .llm_interface import LLMInterface
from typing import Dict, List

logger = logging.getLogger(__name__)

class OllamaLLM(LLMInterface):
    def __init__(self):
        logger.info("Initializing Ollama")

        logger.debug("Ollama client initialized")

    def categorize_bookmarks(self, bookmarks: List[Dict], existing_categories: List[str], is_first_iteration: bool) -> Dict[int, str]:
        logger.info(f"Categorizing {len(bookmarks)} bookmarks")
        bookmark_info = "\n".join([f"{i+1}. Title: {b['title']}\nURL: {b['url']}\nDescription: {b['description']}" 
                                   for i, b in enumerate(bookmarks)])
        
        if is_first_iteration and not existing_categories:
            logger.debug("First iteration with no existing categories")
            prompt = f"""
            ### Task:
            Categorize the given bookmarks into appropriate categories.

            ### Instructions:

            1. Read each bookmark's title, URL, and description carefully.
            2. Create relevant, high-level categories for these bookmarks.
            3. Avoid categorizing bookmarks as "Other" if you can.
            4. Assign each bookmark to one of your created categories.
            5. The category name should not include any labels, descriptions, or additional details.
            6. Provide your response as a JSON object where:
               - Keys are the bookmark IDs (1, 2, 3, etc.)
               - Values are the assigned category names
            7. Ensure every bookmark is categorized.

            ### Bookmarks:
            {bookmark_info}

            ### Your categorization:
            """
        else:
            logger.debug(f"Using {len(existing_categories)} existing categories")
            categories_list = ', '.join(existing_categories)
            prompt = f"""
            ### Task:
            Categorize the given bookmarks into the provided categories or create new ones if necessary.

            ### Instructions:

            1. Read each bookmark's title, URL, and description carefully.
            2. Assign each bookmark to the most appropriate category from the list provided.
            3. Avoid categorizing bookmarks as "Other" if you can.
            4. If no existing category fits well, create a new, relevant high-level category.
            5. The category name should not include any labels, descriptions, or additional details.
            6. Prefer using existing categories when appropriate.
            7. Provide your response as a JSON object where:
               - Keys are the bookmark IDs (1, 2, 3, etc.)
               - Values are the assigned category names
            8. Ensure every bookmark is categorized.

            ### Bookmarks:
            {bookmark_info}

            ### Existing categories:
            {categories_list}

            ### Your categorization:
            """

        try:
            response_content: ChatResponse = chat(model='llama3.2', messages=[
            {
                {"role": "system", "content": "You are an AI assistant that accurately categorizes batches of bookmarks, creating or using provided categories as instructed."},
                {"role": "user", "content": prompt}
            },
            ])

            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    categorization = json.loads(json_match.group())
                    logger.info(f"Successfully categorized {len(categorization)} bookmarks")
                    return {int(k): v for k, v in categorization.items()}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}", exc_info=True)
                    raise ValueError("Failed to parse LLM response as JSON")
            else:
                logger.error("No JSON object found in LLM response")
                raise ValueError("No JSON object found in LLM response")
        except Exception as e:
            logger.error(f"Error during bookmark categorization: {e}", exc_info=True)
            raise