# data files
set (${PROJECT_NAME}_data
    SKA1_low_ecef
        SKA1_mid_ecef
)

# Copy data files
install(
    FILES
    ${${PROJECT_NAME}_data}
    DESTINATION
    data
)
