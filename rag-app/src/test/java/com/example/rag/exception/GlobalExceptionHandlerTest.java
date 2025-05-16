package com.example.rag.exception;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import com.example.rag.WebController;
import com.example.rag.service.RagService;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(WebController.class)
public class GlobalExceptionHandlerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RagService ragService;

    @Test
    public void testQuestionRequiredException() throws Exception {
        // Configure mock service to throw exception
        when(ragService.askQuestion(any())).thenThrow(new QuestionRequiredException());

        // Execute and verify
        mockMvc.perform(get("/ask")
                .param("question", "test")
                .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("Question is required"));
    }

    @Test
    public void testGeneralException() throws Exception {
        // Configure mock service to throw general exception
        when(ragService.askQuestion(any())).thenThrow(new RuntimeException("Something went wrong"));

        // Execute and verify
        mockMvc.perform(get("/ask")
                .param("question", "test")
                .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.error").value("An error occurred"));
    }
}
