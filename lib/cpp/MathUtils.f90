! Fortran code for the efficient computation of the Wigner-3j symbols.
! This code is taken from the CAMB source code, with slight modifications
! to make it work from only a single file, and the removal of any MPI functions.

! The CAMB license can be found at https://github.com/cmbant/CAMB/blob/master/LICENCE.txt
! Copyright (c) Antony Lewis

    
    module Precision
    implicit none

    integer, parameter :: dl = KIND(1.d0)
    integer, parameter :: sp = KIND(1.0)
    end module Precision
    
    module MathUtils
    use precision
    implicit none

    interface
    FUNCTION obj_function(obj, x)
    use precision
    class(*) :: obj
    real(dl) :: x, obj_function
    END FUNCTION  obj_function
    end interface

    contains


    subroutine GetThreeJs(thrcof,l2in,l3in,m2in,m3in) bind(C)
    !Recursive evaluation of 3j symbols. Does minimal error checking on input parameters.
    !! use MpiUtils, only : MpiStop
    implicit none
    integer, parameter :: dl = KIND(1.d0)
    integer, intent(in) :: l2in,l3in, m2in,m3in
    real(dl), dimension(*) :: thrcof
    INTEGER, PARAMETER :: i8 = selected_int_kind(18)
    integer(i8) :: l2,l3,m2,m3
    integer(i8) :: l1, m1, l1min,l1max, lmatch, nfin, a1, a2

    real(dl) :: newfac, oldfac, sumfor, c1,c2,c1old, dv, denom, x, sum1, sumuni
    real(dl) :: x1,x2,x3, y,y1,y2,y3,sum2,sumbac, ratio,cnorm, sign1, thresh
    integer i,ier, index, nlim, sign2
    integer nfinp1,nfinp2,nfinp3, lstep, nstep2,n
    real(dl), parameter :: zero = 0._dl, one = 1._dl
    real(dl), parameter ::  tiny = 1.0d-30, srtiny=1.0d-15, huge = 1.d30, srhuge = 1.d15

    ! routine to generate set of 3j-coeffs (l1,l2,l3\\ m1,m2,m3)

    ! by recursion from l1min = max(abs(l2-l3),abs(m1))
    !                to l1max = l2+l3
    ! the resulting 3j-coeffs are stored as thrcof(l1-l1min+1)

    ! to achieve the numerical stability, the recursion will proceed
    ! simultaneously forwards and backwards, starting from l1min and l1max
    ! respectively.
    !
    ! lmatch is the l1-value at which forward and backward recursion are matched.
    !
    ! ndim is the length of the array thrcof
    !
    ! ier = -1 for all 3j vanish(l2-abs(m2)<0, l3-abs(m3)<0 or not integer)
    ! ier = -2 if possible 3j's exceed ndim
    ! ier >= 0 otherwise

    l2=l2in
    l3=l3in
    m2=m2in
    m3=m3in
    newfac = 0
    lmatch = 0
    m1 = -(m2+m3)

    ! check relative magnitude of l and m values
    ier = 0

    if (l2 < abs(m2) .or. l3 < m3) then
        ier = -1
        ! call MpiStop('error ier = -1')
        return
    end if

    ! limits for l1
    l1min = max(abs(l2-l3),abs(m1))
    l1max = l2+l3

    if (l1min >= l1max) then
        if (l1min/=l1max) then
            ier = -1
            ! call MpiStop('error ier = -1')
            return
        end if

        ! reached if l1 can take only one value, i.e.l1min=l1max
        thrcof(1) = (-1)**abs(l2+m2-l3+m3)/sqrt(real(l1min+l2+l3+1,dl))
        return

    end if

    nfin = l1max-l1min+1

    ! starting forward recursion from l1min taking nstep1 steps
    l1 = l1min
    thrcof(1) = srtiny
    sum1 = (2*l1 + 1)*tiny

    lstep = 1

30  lstep = lstep+1
    l1 = l1+1

    oldfac = newfac
    a1 = (l1+l2+l3+1)*(l1-l2+l3)*(l1+l2-l3)
    a2 = (l1+m1)*(l1-m1)*(-l1+l2+l3+1)
    newfac = sqrt(a2*real(a1,dl))
    if (l1 == 1) then
        !IF L1 = 1  (L1-1) HAS TO BE FACTORED OUT OF DV, HENCE
        c1 = -(2*l1-1)*l1*(m3-m2)/newfac
    else

        dv = -l2*(l2+1)*m1 + l3*(l3+1)*m1 + l1*(l1-1)*(m3-m2)
        denom = (l1-1)*newfac

        if (lstep > 2) c1old = abs(c1)
        c1 = -(2*l1-1)*dv/denom

    end if

    if (lstep<= 2) then

        ! if l1=l1min+1 the third term in the recursion eqn vanishes, hence
        x = srtiny*c1
        thrcof(2) = x
        sum1 = sum1+tiny*(2*l1+1)*c1*c1
        if(lstep==nfin) then
            sumuni=sum1
            go to 230
        end if
        goto 30

    end if

    c2 = -l1*oldfac/denom

    ! recursion to the next 3j-coeff x
    x = c1*thrcof(lstep-1) + c2*thrcof(lstep-2)
    thrcof(lstep) = x
    sumfor = sum1
    sum1 = sum1 + (2*l1+1)*x*x
    if (lstep/=nfin) then

        ! see if last unnormalised 3j-coeff exceeds srhuge
        if (abs(x) >= srhuge) then

            ! REACHED IF LAST 3J-COEFFICIENT LARGER THAN SRHUGE
            ! SO THAT THE RECURSION SERIES THRCOF(1), ... , THRCOF(LSTEP)
            ! HAS TO BE RESCALED TO PREVENT OVERFLOW

            ier = ier+1
            do i = 1, lstep
                if (abs(thrcof(i)) < srtiny) thrcof(i)= zero
                thrcof(i) = thrcof(i)/srhuge
            end do

            sum1 = sum1/huge
            sumfor = sumfor/huge
            x = x/srhuge

        end if

        ! as long as abs(c1) is decreasing, the recursion proceeds towards increasing
        ! 3j-valuse and so is numerically stable. Once an increase of abs(c1) is
        ! detected, the recursion direction is reversed.

        if (c1old > abs(c1)) goto 30

    end if !lstep/=nfin

    ! keep three 3j-coeffs around lmatch for comparison with backward recursion

    lmatch = l1-1
    x1 = x
    x2 = thrcof(lstep-1)
    x3 = thrcof(lstep-2)
    nstep2 = nfin-lstep+3

    ! --------------------------------------------------------------------------
    !
    ! starting backward recursion from l1max taking nstep2 stpes, so that
    ! forward and backward recursion overlap at 3 points
    ! l1 = lmatch-1, lmatch, lmatch+1

    nfinp1 = nfin+1
    nfinp2 = nfin+2
    nfinp3 = nfin+3
    l1 = l1max
    thrcof(nfin) = srtiny
    sum2 = tiny*(2*l1+1)

    l1 = l1+2
    lstep=1

    do
        lstep = lstep + 1
        l1= l1-1

        oldfac = newfac
        a1 = (l1+l2+l3)*(l1-l2+l3-1)*(l1+l2-l3-1)
        a2 = (l1+m1-1)*(l1-m1-1)*(-l1+l2+l3+2)
        newfac = sqrt(a1*real(a2,dl))

        dv = -l2*(l2+1)*m1 + l3*(l3+1)*m1 +l1*(l1-1)*(m3-m2)

        denom = l1*newfac
        c1 = -(2*l1-1)*dv/denom
        if (lstep <= 2) then

            ! if l2=l2max+1, the third term in the recursion vanishes

            y = srtiny*c1
            thrcof(nfin-1) = y
            sumbac = sum2
            sum2 = sum2 + tiny*(2*l1-3)*c1*c1

            cycle

        end if

        c2 = -(l1-1)*oldfac/denom

        ! recursion to the next 3j-coeff y
        y = c1*thrcof(nfinp2-lstep)+c2*thrcof(nfinp3-lstep)

        if (lstep==nstep2) exit

        thrcof(nfinp1-lstep) = y
        sumbac = sum2
        sum2 = sum2+(2*l1-3)*y*y

        ! see if last unnormalised 3j-coeff exceeds srhuge
        if (abs(y) >= srhuge) then

            ! reached if 3j-coeff larger than srhuge so that the recursion series
            ! thrcof(nfin),..., thrcof(nfin-lstep+1) has to be rescaled to prevent overflow

            ier=ier+1
            do i = 1, lstep
                index=nfin-i+1
                if (abs(thrcof(index)) < srtiny) thrcof(index)=zero
                thrcof(index) = thrcof(index)/srhuge
            end do

            sum2=sum2/huge
            sumbac=sumbac/huge

        end if

    end do

    ! the forward recursion 3j-coeffs x1, x2, x3 are to be matched with the
    ! corresponding backward recursion vals y1, y2, y3

    y3 = y
    y2 = thrcof(nfinp2-lstep)
    y1 = thrcof(nfinp3-lstep)

    ! determine now ratio such that yi=ratio*xi (i=1,2,3) holds with minimal error

    ratio = (x1*y1+x2*y2+x3*y3)/(x1*x1+x2*x2+x3*x3)
    nlim = nfin-nstep2+1

    if (abs(ratio) >= 1) then

        thrcof(1:nlim) = ratio*thrcof(1:nlim)
        sumuni = ratio*ratio*sumfor + sumbac

    else

        nlim = nlim+1
        ratio = 1/ratio
        do n = nlim, nfin
            thrcof(n) = ratio*thrcof(n)
        end do
        sumuni = sumfor + ratio*ratio*sumbac

    end if
    ! normalise 3j-coeffs

230 cnorm = 1/sqrt(sumuni)

    ! sign convention for last 3j-coeff determines overall phase

    sign1 = sign(one,thrcof(nfin))
    sign2 = (-1)**(abs(l2+m2-l3+m3))
    if (sign1*sign2 <= 0) then
        cnorm = -cnorm
    end if
    if (abs(cnorm) >= one) then
        thrcof(1:nfin) = cnorm*thrcof(1:nfin)
        return
    end if

    thresh = tiny/abs(cnorm)

    do n = 1, nfin
        if (abs(thrcof(n)) < thresh) thrcof(n) = zero
        thrcof(n) = cnorm*thrcof(n)
    end do
    return

    end subroutine GetThreeJs
   

    end module MathUtils
