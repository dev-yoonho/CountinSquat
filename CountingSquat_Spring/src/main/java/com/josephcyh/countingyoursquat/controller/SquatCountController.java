package com.josephcyh.countingyoursquat.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class SquatCountController {

    private int squatCount = 0;

    // Flask에서 스쿼트 개수를 받는 엔드포인트
    @PostMapping("/receiveSquatCount")
    public ResponseEntity<String> receiveSquatCount(@RequestBody Map<String, Object> squatData) {
        try {
            squatCount = (Integer) squatData.get("squatCount");
            String videoFileName = (String) squatData.get("fileName");

            System.out.println("Received squat count: " + squatCount + " for file: " + videoFileName);

            return ResponseEntity.ok("Squat count received successfully");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Failed to receive squat count");
        }
    }

    // 사용자에게 총 스쿼트 개수를 보여주는 엔드포인트
    @GetMapping("/showSquatCount")
    public ModelAndView showSquatCount() {
        ModelAndView mav = new ModelAndView("squatResult");
        mav.addObject("squatCount", squatCount);
        return mav;
    }
}
