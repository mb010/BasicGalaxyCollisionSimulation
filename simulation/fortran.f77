C	Defining the variuos arrays to hold our N-body systems information
DIMENSION X(3,50),X0(3,50),X0DOT(3,50),T0(50),BODY(50),STEP(50),F(3,50),FDOT(3,50),D1(3,50),D2(3,50),D3(3,50),T1(50),T2(50),T3(50),A(17),F1(3),F1DOT(3),F3DOT(3)
DATA TIME,TNEXT,NSTEPS /0.0,0.0,0/

C      Inputing the starting conditions for the system
READ (5,*) N,ETA,DELTAT,TCRIT,EPS2

Do 1 I = 1,N
    1 READ (5,*) BODY8I),(X0(K,I)),K=1,3),(X0DOT(K,I),K=1,3)

C      OBTAIN TOTAL FORCE AND FIRST DERIVATIVE FOR EACH BODY
DO 20 I = 1,N
    DO 2 K = 1,3
        F(K,I) = 0.0
        FDOT(K,I) = 0.0
        D2(K,I) = 0.0
        2 D3(K,I) = 0.0
    DO 10 J = 1,N
        IF (J.EQ.I) GO TO 10
        DO 5 K = 1,3
            A(K) = X0(K,J) - X0(K,I)
            5 A(K+3) = X0DOT(K,J) - X0DOT(K,I)
        A(7) = 1.0/(A(1)**2 + A(2)**2) + A(3)**2 + EPS2)
        A(8) = BODY(J)*A(7)*SQRT (A(7))
        A(9) = 3.0*(A(1)*A(4) + A(2)*A(5) + A(3)*A(6))*A(7)
        Do 8 K = 1,3
            F(K,I) = F(K,I) + A(K)*A(8)
            8 FDOT(K,I) = FDOT(K,I) + (A(K+3) - A(K)*A(9))*A(8)
        10 CONTINUE
    20 CONTINUE

C      FORM SECOND AND THIRD DERIVATIVE
DO 40 I = 1,N
    DO 30 J = 1,N
        IF (J.EQ.I) GO TO 30
        DO 25 K = 1,3
            A(K) = X0(K,J) - X0(K,I)
            A(K+3) = X0DOT(K,J) - X0DOT(K,I)
            A(K+6) = F(K,J) - F(K,I)
            25 A(K+9) = FDOT(K,J) - FDOT(K,I)
        A(13) = 1.0/(A(1)**2 + A(2)**2 + A(3)**2 + EPS2)
        A(14) = BODY(J)*A(13)*SQRT (A(13))
        A(15) = (A(1)*A(4) + A(2)*A(5) + A(3)*A(6))*A(13)
        A(16) = (A(4)**2 + A(5)**2 + A(6)**2 + A(1)*A(7) + A(2)*A(8) + A(3)*A(9))*A(13) + A(15)**2
        A(17) = (9.0*(A(4)*A(7) + A(5)*A(8) + A(6)*A(9)) + 3.0*(A(1)*A(10) + A(2)*A(11) + A(3)*A(12)))*A(13) + A(15)*(9.*A(16) -12.*A(15)**2)
        DO 28 K = 1,3
            F1DOT(K) = A(K+3) - 3.0*A(15)*A(K)
            F2DOT(K) = (A(K+6) - 6.0*A(15)*F1DOT(K) - 3.0*A(16)*A(K))*A(14)
            F3DOT(K) = (A(K+9) - 9.0*A(16)+F1DOT(K) - A(17)*A(K))*A(14)
            D2(K,I) = D2(K,I) + F2DOT(K)
            28 D3(K,I) = D3(K,I) + F3DOT(K) - 9.0*A(15)*F2DOT(K)
        30 CONTINUE
    40 CONTINUE

C      INITIALISE INTEGRATION STEPS AND CONVERT TO FORCE DIFFERENCES.
DO 50 I = 1,N
    STEP(I) = SQRT (ETA*SQRT ((F(1,I)**2 + F(2,I)**2 + F(3,I)**2)/(D2(1,I)**2 + D2(2,I)**2 + D2(3,I)**2)))
    T0(I) = TIME
    T1(I) 0 TIME -STEP(I)
    T2(I) = TIME - 2.O*STEP(I)
    T3(I) = TIME - 3.O*STEP(I)
    DO 45 K = 1,3
        D1(K,I) = (D3(K,I)*STEP(I)/6.0 - 0.5*D2(K,I))*STEP(I) + FDOT(K,I)
        D2(K,I) = 0.5*D2(K,I) - 0.5*D3(K,I)+STEP(I)
        D3(K,I) = D3(K,I)/6.0
        F(K,I) = FDOT(K,I)/2.0
        45 FDOT(K,I) = FDOT(K,I)/6.0
    50 CONTINUE

C      ENERGY CHECK AND OUTPUT
100 E = 0.0
DO 110 I=1,N
    DT = TNEXT - T0(I)
    DO 101 K = 1,3
        F2DOT(K) = D3(K,I)*((T0(I) - T1(I)) + (T0(I) - T2(I))) + D2(K,I)
        x(K,I) = ((((0.05*D3(K,I)*DT + F2DOT(K)/12.0)*DT + FDOT(K,I))*DT + F(K,I))*DT + X0DOT(K,I))*DT + X0(K,I)
        101 A(K) = (((0.25*D3(K,I)*DT + F2DOT(K)/3.0)*DT + 3.0*FDOT(K,I)*DT + 2.0*F(K,I)*DT + X0DOT(K,I)
    WRITE (6,105) I, BODY(I) , STEP(I) , (X(K,I),K=1,3) , (A(K),K=1,3)
        105 FORMAT  (1H , I10,F10.2,F12.4,3X,3F10.2,3X,3F10.2)
    110 E = E +0.5*BODY(I)*(A(1)**2 + A(2)**2 + A(3)**2)
DO 130 I = 1,N
    DO 120 J = 1,N
        IF (J.EQ.I) GO TO 120
        E = E - 0.5*BODY(I)*BODY(J)/SQRT ((X(1,I) - X(I,J))**2 + (X(2,I) - X(2,J))**2 + (X(3,I) - X(3,J))**2 + EPS2)
        120 CONTINUE
    130 CONTINUE
WRITE (6,140) TNEXT,NSTEPS,E
    140 FORMAT  (1H0,5X,'TIME =',F7.2,' STEPS =',16,' ENERGY =',F10.4,/)
IF  (TIME.GT.TCRIT) STOP
TNEXT = TNEXT + DELTAT

C       FIND NEXT BODY TO BE ADVANCED AND SET NEW TIME
200 TIME = 1.0E+10
DO 210 J = 1,N
    IF (TIME.GT.T0(J) + STEP(J)) I = J
    IF (TIME.GT.T0(J) + STEP(J)) TIME = T0(J) + STEP(J)
    210 CONTINUE

C      PREDICT ALL COORDINATES TO FIRST ORDER IN FORCE DERIVATIVE
DO 220 J = 1,N
    S = TIME -T0(J)
    X(1,J) = ((FDOT(1,J)*S + F(1,J))*S + X0DOT(1,J))*S + X0(1,J)
    X(2,J) = ((FDOT(2,J)*S + F(2,J))*S + X0DOT(2,J))*S + X0(2,J)
    220 X(3,J) = ((FDOT(3,J)*S + F(3,J))*S + X0DOT(3,J))*S + X0(3,J)

C      INCLUDE SECOND AND THUIRD ORDER AND OBTAIN THE VELOCITY
DT = TIME - T0(I)
DO 230 K = 1,3
    F2DOT(K) = D3(K,I)*((T0(I) - T1(I)) + (T0(I) - T2(I))) + D2(K,I)
    X(K,I) = (0.05*D3(K,I)*DT + F2DOT(K)/12.0)*DT**4 + X(K,I)
    X0DOT(K,I) = (((0.25*D3(K,I)*DT + F2DOT(K)/3.0)*DT + 3.0*FDOT(K,I))*DT + 2.0*F(K,I))*DT + X0DOT(K,I)
    230 F1(K) = 0.0

C      OBTAIN THE CURRENT FRCE ON THE i'th BODY
DO 240 J = 1,N
    IF (J.EQ.I) GO TO 240
    A(1) = X(1,J) - X(1,I)
    A(2) = X(2,J) - X(2,I)
    A(3) = X(3,J) - X(3,I)
    A(4) = 1.0/(A(1)**2 + A(2)**2 +A(3)**2 +EPS2)
    A(5) = BODY(J)*A(4)*SQRT (A(4))
    F1(1) = F1(1) + A(1)*A(5)
    F1(2) = F1(2) + A(2)*A(5)
    F1(3) = F1(3) + A(3)*A(5)
    240 CONTINUE

C      SET TIME INTERVALS FOR NEW DIFFERENCES AND UPDATE THE TIMES
DT1 = TIME - T1(I)
DT2 = TIME - T2(I)
DT3 = TIME - T3(I)
T1PR = T0(I) - T1(I)
T2PR = T0(I) - T2(I)
T3PR = T0(I) - T3(I)
T3(I) = T2(I)
T2(I) = T1(I)
T1(I) = T0(I)
T0(I) = TIME

C      FORM NEW DIFFERENCES AND INCLUDE FOURTH-ORDER SEMI ITERATION
DO 250 K = 1,3
    A(K) = (F1(K) - 2.0*F(K,I))/DT
    A(K+3) = (A(K) - D1(K,I))/DT1
    A(K+6) = (A(K+3) - D2(K,I))/DT2
    A(K+9) = (A(K+6) - D3(K,I))/DT3
    D1(K,I) = A(K)
    D2(K,I) = A(K+3)
    D3(K,I) = A(K+6)
    F1DOT(K) = T1PR*T2PR*T3PR*A(K+9)
    F2DOT(K) = (T1PR*T2PR + T3PR*(T1PR + T2PR))*A(K+9)
    F3DOT(K) = (T1PR + T2PR + T3PR)*A(K+9)
    X0(K,I) = (((A(K+9)*DT/30.0 + 0.05*F3DOT(K))*DT + F2DOT(K)/12.0)*DT + F1DOT(K)/6.0)*DT**3 + X(K,I)
    250 X0DOT(K,I) = (((0.2*A(K+9)*DT + 0.25*F3DOT(K))*DT + F2DOT(K)/3.0)*DT + 0.5*F1DOT(K))*DT**2 + X0DOT(K,I)

C      SCALE F AND FDOT BY FACTORIALS AND SET NEW INTEGRATION STEP
DO 260 K = 1,3
    F(K,I) = 0.5*F1(K)
    FDOT(K,I) = ((D3(K,I)*DT1 + D2(K,I))*DT + D1(K,I))/6.0
    260 F2DOT(K) = 2.0*(D3(K,I)*(DT + DT1) + D2(K,I))
STEP(I) = SQRT (ETA*SQRT ((F1(1)**2 + F1(2)**2 + F1(3)**2)/(F2DOT(1)**2 + F2DOT(2)**2 + F2DOT(3)**2)))
NSTEPS = NSTEPS + 1

IF (TIME -TNEXT) 200,100,100 
C	IF(Arithmetic) see https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vn9p/index.html
END
